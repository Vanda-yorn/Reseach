import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import Tuple, Dict, List
from SWIN import SwinEncoder, SwinDecoder

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, power_actions, bit_actions):
        super(DQNNetwork, self).__init__()
        self.num_power_actions = len(power_actions)
        self.num_bit_actions = len(bit_actions)

        # Input normalization
        self.input_norm = nn.LayerNorm(state_dim)

        # Deeper shared layers with residual connections
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)

        # Value stream
        self.value_fc = nn.Linear(512, 256)
        self.value_ln = nn.LayerNorm(256)
        self.value = nn.Linear(256, 1)

        # Advantage streams for power and bits
        self.power_fc = nn.Linear(512, 256)
        self.power_ln = nn.LayerNorm(256)
        self.power_advantage = nn.Linear(256, self.num_power_actions)

        self.bits_fc = nn.Linear(512, 256)
        self.bits_ln = nn.LayerNorm(256)
        self.bits_advantage = nn.Linear(256, self.num_bit_actions)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.input_norm(state)

        # Shared layers with residual connections
        identity = self.fc1(x)
        x = torch.relu(self.ln1(identity))
        x = torch.relu(self.ln2(self.fc2(x))) + identity
        x = torch.relu(self.ln3(self.fc3(x))) + x

        # Value stream
        v = torch.relu(self.value_ln(self.value_fc(x)))
        value = self.value(v)

        # Power advantage stream
        pa = torch.relu(self.power_ln(self.power_fc(x)))
        power_advantage = self.power_advantage(pa)
        power_q = value + (power_advantage - power_advantage.mean(dim=1, keepdim=True))

        # Bits advantage stream
        ba = torch.relu(self.bits_ln(self.bits_fc(x)))
        bits_advantage = self.bits_advantage(ba)
        bits_q = value + (bits_advantage - bits_advantage.mean(dim=1, keepdim=True))

        return power_q, bits_q


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        # Set initial priority to max priority in buffer or 1 if buffer is empty
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        try:
            # Ensure we have enough samples
            if len(self.buffer) < batch_size:
                return random.sample(list(self.buffer), len(self.buffer))
            
            # Calculate probabilities
            priorities = np.array(self.priorities)
            # Add small constant to avoid zero probabilities
            probabilities = (priorities + 1e-6) / (np.sum(priorities) + 1e-6 * len(priorities))
            
            # Verify probabilities sum to 1
            probabilities = probabilities / np.sum(probabilities)
            
            # Sample indices
            indices = np.random.choice(
                len(self.buffer), 
                min(batch_size, len(self.buffer)), 
                p=probabilities,
                replace=True
            )
            
            # Return sampled experiences
            return [self.buffer[idx] for idx in indices]
            
        except Exception as e:
            print(f"Error in sampling: {str(e)}")
            # Fallback to random sampling
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class SemanticDQNAgent:
    def __init__(
        self,
        state_dim,
        power_actions,
        bit_actions,
        learning_rate=0.00005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        buffer_size=500000,
        batch_size=256,
        target_update=50,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.power_actions = power_actions
        self.bit_actions = bit_actions

        self.policy_net = DQNNetwork(state_dim, power_actions, bit_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, power_actions, bit_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Using Adam optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        self.replay_buffer = ReplayBuffer(buffer_size)

        # Initialize reward tracking
        self.reward_deque = deque(maxlen=1000)
        self.reward_scale = 1.0

        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        # Temperature parameter for softmax exploration
        self.temperature = 1.0

    def update_reward_scale(self, reward):
        self.reward_deque.append(abs(reward))
        if len(self.reward_deque) >= 100:
            scale = 1.0 / (np.mean(self.reward_deque) + 1e-6)
            self.reward_scale = min(scale, 1e-6)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)

        if random.random() > self.epsilon:
            with torch.no_grad():
                power_q, bits_q = self.policy_net(state_tensor)
                
                # Simple approach: just use argmax
                power_action = power_q.argmax(dim=1).item()
                bits_action = bits_q.argmax(dim=1).item()
                
                # Optional: Add small random chance to pick nearby actions
                if random.random() < 0.1:  # 10% chance
                    power_action = min(max(0, power_action + random.randint(-1, 1)), 
                                    len(self.power_actions) - 1)
                    bits_action = min(max(0, bits_action + random.randint(-1, 1)), 
                                    len(self.bit_actions) - 1)
        else:
            # Simple random exploration
            power_action = random.randrange(len(self.power_actions))
            bits_action = random.randrange(len(self.bit_actions))

        return {
            "power": self.power_actions[power_action],
            "bits": self.bit_actions[bits_action],
        }
    def process_state(self, env_state):
        channel_gains = env_state["channel_gains"].flatten()
        transmission_rates = env_state["transmission_rates"].flatten()
        user_positions = env_state["user_positions"].flatten()
        uav_positions = env_state["uav_positions"].flatten()

        state = np.concatenate([
            channel_gains,
            transmission_rates,
            user_positions,
            uav_positions,
        ])

        # Normalize state values
        state_mean = np.mean(state)
        state_std = np.std(state) + 1e-6
        normalized_state = (state - state_mean) / state_std

        return normalized_state

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        try:
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = list(zip(*transitions))

            # Convert to tensors and enable gradients
            state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
            action_batch = np.array(batch[1])
            reward_batch = torch.FloatTensor(np.array(batch[2])).to(self.device)
            next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
            done_batch = torch.FloatTensor(np.array(batch[4])).to(self.device)

            # Get current Q values (without no_grad since we need gradients)
            current_power_q, current_bits_q = self.policy_net(state_batch)

            # Get next Q values (with no_grad since we don't need gradients for targets)
            with torch.no_grad():
                next_power_q, next_bits_q = self.target_net(next_state_batch)
                next_power_values = next_power_q.max(1)[0]
                next_bits_values = next_bits_q.max(1)[0]

                # Compute target Q values
                power_target = reward_batch + (1 - done_batch) * self.gamma * next_power_values
                bits_target = reward_batch + (1 - done_batch) * self.gamma * next_bits_values

            # Extract action indices
            power_actions = torch.tensor([self.power_actions.index(a["power"]) for a in action_batch], 
                                    device=self.device)
            bits_actions = torch.tensor([self.bit_actions.index(a["bits"]) for a in action_batch], 
                                    device=self.device)

            # Get Q values for taken actions
            power_q_values = current_power_q.gather(1, power_actions.unsqueeze(1)).squeeze(1)
            bits_q_values = current_bits_q.gather(1, bits_actions.unsqueeze(1)).squeeze(1)

            # Compute losses
            criterion = nn.SmoothL1Loss()
            power_loss = criterion(power_q_values, power_target)
            bits_loss = criterion(bits_q_values, bits_target)
            total_loss = power_loss + bits_loss

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()

            # Update target network occasionally
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return total_loss.item()

        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())
            return None


class DIV2KDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in directory: {root}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as a dummy label


@dataclass
class SystemParameters:
    num_users: int = 50
    num_uavs: int = 4
    radius: float = 1000  # meters
    user_height: float = 0  # meters
    uav_height: float = 150  # meters
    bandwidth: float = 1e6  # Hz
    noise_power: float = 1e-9  # Watts
    transmit_power: float = 0.1  # Watts
    carrier_frequency: float = 2.4e9  # Hz
    wavelength: float = 3e8 / 2.4e9
    data_size: int = 1e6  # bits per user
    time_duration: float = 1  # seconds


class SemanticCommunicationEnv:
    def __init__(self, params: SystemParameters = SystemParameters()):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iot_devices = []
        self.uavs = []

        # Add encoder and decoder attributes
        self.encoder = None
        self.decoder = None

        # Define action space
        self.action_space = {
            "power": np.linspace(0.01, 3, 10).tolist(),  # Discrete power levels
            "bits": [int(x) for x in np.linspace(1, 8, 10)],  # Discrete bit levels
        }

        # State tracking
        self.channel_gains = None
        self.transmission_rates = None
        self.current_positions = None

        # Load models
        self.load_models()

        # Initialize environment
        self._initialize_network()

    def _generate_user_positions(self) -> np.ndarray:
        """Generate random user positions within circular area"""
        angles = np.random.uniform(0, 2 * np.pi, self.params.num_users)
        radii = np.sqrt(np.random.uniform(0, self.params.radius**2, self.params.num_users))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def _determine_uav_positions(self, user_positions: np.ndarray) -> np.ndarray:
        """Use K-means clustering to determine UAV positions"""
        kmeans = KMeans(n_clusters=self.params.num_uavs, random_state=42)
        kmeans.fit(user_positions)
        return kmeans.cluster_centers_

    def _initialize_network(self):
        """Initialize network topology"""
        user_positions = self._generate_user_positions()
        uav_positions = self._determine_uav_positions(user_positions)
        self.current_positions = {
            "users": user_positions,
            "uavs": uav_positions,
        }
        self._update_channel_conditions()

    def _calculate_distances(self):
        """Calculate distances between users and UAVs"""
        distances = np.zeros((self.params.num_users, self.params.num_uavs))
        for i in range(self.params.num_users):
            for j in range(self.params.num_uavs):
                dx = self.current_positions["users"][i, 0] - self.current_positions["uavs"][j, 0]
                dy = self.current_positions["users"][i, 1] - self.current_positions["uavs"][j, 1]
                dz = self.params.uav_height
                distances[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)
        return distances

    def _update_channel_conditions(self):
        """Update channel conditions for all user-UAV pairs"""
        distances = self._calculate_distances()

        self.channel_gains = np.zeros((self.params.num_users, self.params.num_uavs))
        self.transmission_rates = np.zeros((self.params.num_users, self.params.num_uavs))

        for i in range(self.params.num_users):
            for j in range(self.params.num_uavs):
                # Path loss
                path_loss = (self.params.wavelength / (4 * np.pi * distances[i, j])) ** 2
                fading = np.random.rayleigh(scale=1.0)
                self.channel_gains[i, j] = path_loss * fading

                # Calculate rate using SINR
                interference = 0.01
                sinr = (self.params.transmit_power * self.channel_gains[i, j]) / (
                    self.params.noise_power + interference
                )
                self.transmission_rates[i, j] = self.params.bandwidth * np.log2(1 + sinr)

    def calculate_ssim(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate SSIM between original and transmitted features"""
        c1, c2 = 0.01, 0.03
        mu_x, mu_y = np.mean(x), np.mean(y)
        sigma_x, sigma_y = np.std(x), np.std(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))

        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
        )
        return ssim

    def load_models(self):
        """Load pre-trained Swin Transformer models"""
        try:
            # Initialize models first
            self.encoder = SwinEncoder()  # Use your imported SwinEncoder class
            self.decoder = SwinDecoder()  # Use your imported SwinDecoder class
            
            encoder_path = "./saved_models/swin_transformer/encoder_div2k.pth"
            decoder_path = "./saved_models/swin_transformer/decoder_div2k.pth"

            print(f"Loading encoder from {encoder_path}")
            encoder_checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=True)

            print(f"Loading decoder from {decoder_path}")
            decoder_checkpoint = torch.load(decoder_path, map_location=self.device, weights_only=True)

            # Load state dictionaries
            if isinstance(encoder_checkpoint, dict):
                if 'state_dict' in encoder_checkpoint:
                    self.encoder.load_state_dict(encoder_checkpoint['state_dict'])
                elif 'model' in encoder_checkpoint:
                    self.encoder.load_state_dict(encoder_checkpoint['model'])
                else:
                    self.encoder.load_state_dict(encoder_checkpoint)

            if isinstance(decoder_checkpoint, dict):
                if 'state_dict' in decoder_checkpoint:
                    self.decoder.load_state_dict(decoder_checkpoint['state_dict'])
                elif 'model' in decoder_checkpoint:
                    self.decoder.load_state_dict(decoder_checkpoint['model'])
                else:
                    self.decoder.load_state_dict(decoder_checkpoint)

            # Move models to device and set to evaluation mode
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)
            self.encoder.eval()
            self.decoder.eval()

            print("Encoder state dict keys:", self.encoder.state_dict().keys())
            # Print first layer's weight shape
            first_layer = next(iter(self.encoder.state_dict().values()))
            print("First layer shape:", first_layer.shape)
            print("Successfully loaded Swin Transformer models")
            return True
            

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            # Initialize fresh models if loading fails
            self.encoder = SwinEncoder()
            self.decoder = SwinDecoder()
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)
            self.encoder.eval()
            self.decoder.eval()
            return False

    def process_semantic_features(self, image, user_id):
        """Process image through Swin Transformer encoder"""
        with torch.no_grad():
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()

            image = image.to(self.device)
            if image.dim() == 3:
                image = image.unsqueeze(0)

            semantic_features = self.encoder(image,user_id)
            return semantic_features

    def step(self, iot_id: int, action: Dict, semantic_input=None) -> Tuple[Dict, float, bool, Dict]:
        transmit_power = action["power"]
        num_bits = action["bits"]

        # Find best UAV based on transmission rate
        uav_id = np.argmax(self.transmission_rates[iot_id])
        rate = self.transmission_rates[iot_id, uav_id]

        # Calculate latency
        latency = num_bits / rate if rate > 0 else float("inf")

        if self.encoder is not None and self.decoder is not None and semantic_input is not None:
            with torch.no_grad():
                try:
                    # Prepare input
                    if semantic_input.dim() == 3:
                        semantic_input = semantic_input.unsqueeze(0)
                    
                    # Ensure input size matches IMG_SIZE from SWIN model (128x128)
                    if semantic_input.shape[-1] != 128 or semantic_input.shape[-2] != 128:
                        semantic_input = transforms.functional.resize(semantic_input, (128, 128))
                    
                    semantic_input = semantic_input.to(self.device)

                    # Map iot_id to range [0, NUM_USERS-1]
                    mapped_iot_id = iot_id % 5  # Since NUM_USERS = 5
                    iot_id_tensor = torch.tensor([mapped_iot_id], device=self.device)

                    # Process through encoder
                    original_features = self.encoder(semantic_input, iot_id_tensor)

                    # Channel simulation
                    noise_power = self.params.noise_power / transmit_power
                    channel_noise = torch.randn_like(original_features) * np.sqrt(noise_power)
                    transmitted_features = original_features + channel_noise

                    # Reconstruction
                    reconstructed = self.decoder(transmitted_features)

                    # Calculate SSIM
                    ssim = self.calculate_ssim(
                        original_features.cpu().numpy(),
                        transmitted_features.cpu().numpy()
                    )

                except Exception as e:
                    print(f"Error in semantic processing: {str(e)}")
                    print(f"Input shape: {semantic_input.shape}")
                    print(f"IoT ID: {mapped_iot_id}")
                    # Fallback to basic simulation
                    original_features = np.random.randn(num_bits)
                    noise_power = self.params.noise_power / transmit_power
                    channel_noise = np.random.normal(0, np.sqrt(noise_power), num_bits)
                    transmitted_features = original_features + channel_noise
                    reconstructed = None
                    ssim = self.calculate_ssim(original_features, transmitted_features)

        # Normalize latency to [0, 1] range
        max_latency = self.params.time_duration
        normalized_latency = min(latency / max_latency, 1.0)

        # Calculate reward components
        ssim_reward = ssim
        latency_penalty = normalized_latency
        power_efficiency = 1 - (transmit_power / max(self.action_space["power"]))

        # Combine rewards with weighting
        lambda_ssim = 1.0
        lambda_latency = 1.0
        lambda_power = 0.5

        reward = (
            lambda_ssim * ssim_reward
            - lambda_latency * latency_penalty
            + lambda_power * power_efficiency
        )

        # Scale reward for better learning dynamics
        reward = reward * 100

        # Update channel conditions
        self._update_channel_conditions()
        next_state = self._get_state()

        # Determine if episode is done
        done = latency > self.params.time_duration or ssim < 0.3

        # Prepare info dictionary
        info = {
            "transmission_rate": rate,
            "latency": latency,
            "ssim": ssim,
            "normalized_latency": normalized_latency,
            "power_efficiency": power_efficiency,
            "assigned_uav": uav_id,
            "reconstructed_features": transmitted_features if reconstructed is None else reconstructed,
            "original_features": original_features,
        }

        return next_state, reward, done, info

    def reset(self) -> Dict:
        """Reset environment and return initial state"""
        self._initialize_network()
        return self._get_state()

    def _get_state(self) -> Dict:
        """Get current state of the environment"""
        return {
            "channel_gains": self.channel_gains,
            "transmission_rates": self.transmission_rates,
            "user_positions": self.current_positions["users"],
            "uav_positions": self.current_positions["uavs"],
        }

    def visualize_network(self):
        """Visualize network topology"""
        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.current_positions["users"][:, 0],
            self.current_positions["users"][:, 1],
            color="blue",
            label="Users",
            alpha=0.7,
        )
        plt.scatter(
            self.current_positions["uavs"][:, 0],
            self.current_positions["uavs"][:, 1],
            color="red",
            label="UAVs",
            alpha=0.7,
        )
        plt.legend()
        plt.title("Network Topology")
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def train_semantic_dqn(env, agent, data_loader, num_episodes, max_steps):
    """Train the Semantic DQN agent with image data."""
    episode_rewards = []

    for episode in range(num_episodes):
        try:
            state = env.reset()
            processed_state = agent.process_state(state)
            total_reward = 0
            done = False
            steps = 0

            for batch_idx, (images, _) in enumerate(data_loader):
                if steps >= max_steps or done:
                    break

                for image in images:
                    try:
                        # Debug prints for state
                        # print("\nStep Debug:")
                        # print("Processed state shape:", processed_state.shape)
                        
                        # Get action
                        action = agent.select_action(processed_state)
                        #print("Selected action:", action)
                        
                        # Get random IoT ID
                        iot_id = random.randint(0, env.params.num_users - 1)
                      #  print("IoT ID:", iot_id)

                        # Ensure image is correctly shaped
                        if image.dim() == 3:
                            image = image.unsqueeze(0)
                      #  print("Image shape:", image.shape)

                        # Take step in environment
                        next_state, reward, done, info = env.step(iot_id, action, image)
                      #  print("Reward:", reward)
                       # print("Done:", done)
                        
                        # Process next state
                        processed_next_state = agent.process_state(next_state)
                       # print("Processed next state shape:", processed_next_state.shape)

                        # Store transition
                        agent.replay_buffer.push(
                            processed_state,
                            action,
                            reward,
                            processed_next_state,
                            done
                        )

                        # Train the agent
                        loss = agent.train_step()
                        # if loss is not None:
                        #     print("Training loss:", loss)

                        processed_state = processed_next_state
                        total_reward += reward
                        steps += 1

                        if steps % 10 == 0:
                            print(f"Episode {episode}, Step {steps}")
                            print(f"SSIM: {info.get('ssim', 0):.4f}")
                            print(f"Latency: {info.get('latency', 0):.4f}")

                    except Exception as e:
                        print(f"Error during step iteration: {str(e)}")
                        print(f"Error details: {type(e).__name__}")
                        import traceback
                        print(traceback.format_exc())
                        continue

            episode_rewards.append(total_reward)
            if episode % 10 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}, "
                      f"Epsilon = {agent.epsilon:.4f}")

        except Exception as e:
            print(f"Error during episode {episode}: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())
            continue

    return episode_rewards


def plot_training_results(episode_rewards, window_size=50):
    plt.figure(figsize=(12, 6))
    # plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')

    # Plot moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            range(window_size - 1, len(episode_rewards)),
            moving_avg,
            label=f"Moving Average ({window_size} episodes)",
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


# Modify these parameters in your main script:
if __name__ == "__main__":
    # Create environment
    env = SemanticCommunicationEnv()

    # Set up your image dataset with smaller subset
    dataset_path = r"C:\Users\AINS\Documents\YornVanda\Research_Works\Codes\Test_Code\SWIN_MODEL\div2k\DIV2K_train_HR\DIV2K_train_HR"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = DIV2KDataset(dataset_path, transform=transform)
    # Reduce dataset size - just take 3 images for testing
    subset_indices = np.random.choice(len(dataset), 3, replace=False)
    subset_dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(subset_dataset, batch_size=2, shuffle=True, num_workers=0)

    # Get action spaces from environment
    power_actions = env.action_space["power"]
    bit_actions = env.action_space["bits"]

    # Calculate state dimension
    initial_state = env.reset()
    temp_agent = SemanticDQNAgent(1, power_actions, bit_actions)
    processed_state = temp_agent.process_state(initial_state)
    state_dim = len(processed_state)

    # Create agent with faster learning parameters
    agent = SemanticDQNAgent(
        state_dim=state_dim,
        power_actions=power_actions,
        bit_actions=bit_actions,
        learning_rate=0.0001,  # Increased learning rate
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,  # Faster decay
        buffer_size=50000,    # Smaller buffer
        batch_size=128,       # Smaller batch size
        target_update=10,    # More frequent updates
    )

    # Reduced training parameters
    num_episodes = 500       # Reduced from 1000
    max_steps = 1000        # Reduced from 1000

    print("Starting training...")
    print(f"State dimension: {state_dim}")
    print(f"Number of power actions: {len(power_actions)}")
    print(f"Number of bit actions: {len(bit_actions)}")
    print(f"Device: {agent.device}")

    # Train the agent
    episode_rewards = train_semantic_dqn(env, agent, data_loader, num_episodes, max_steps)

    # Plot results
    plot_training_results(episode_rewards)

    # Save the trained model
    torch.save({
        "policy_net_state_dict": agent.policy_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode_rewards": episode_rewards,
    }, "semantic_dqn_model_test.pth")

    print("Test training completed!")