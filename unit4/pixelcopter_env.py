import math
import sys
import os
from collections import deque

from ple import PLE
from ple.games.base.pygamewrapper import PyGameWrapper
from ple.games.pixelcopter import Block, HelicopterPlayer, Terrain
import pygame
from pygame.constants import K_w, K_s
from gym import spaces

import numpy as np

from collections import deque

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json
import imageio

import tempfile


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, trunacted, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    state, _ = env.reset()
    img = env.render(mode='rgb_array')
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, done, truncated, info = env.step(action) # We directly put next_state = state for recording logic
        img = env.render(mode='rgb_array')
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def push_to_hub(repo_id, model, hyperparameters, eval_env, video_fps=30):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param model: the pytorch model we want to save
    :param hyperparameters: training hyperparameters
    :param eval_env: evaluation environment
    :param video_fps: how many frame per seconds to record our video replay
    """

    _, repo_name = repo_id.split("/")
    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        # Step 2: Save the model
        torch.save(model, local_directory / "model.pt")

        # Step 3: Save the hyperparameters to JSON
        with open(local_directory / "hyperparameters.json", "w") as outfile:
          json.dump(hyperparameters, outfile)

        # Step 4: Evaluate the model and build JSON
        mean_reward, std_reward = evaluate_agent(eval_env,
                                                hyperparameters["max_t"],
                                                hyperparameters["n_evaluation_episodes"],
                                                model)
        # Get datetime
        eval_datetime = datetime.datetime.now()
        eval_form_datetime = eval_datetime.isoformat()

        evaluate_data = {
              "env_id": hyperparameters["env_id"],
              "mean_reward": mean_reward,
              "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
              "eval_datetime": eval_form_datetime,
        }

        # Write a JSON file
        with open(local_directory / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 5: Create the model card
        env_name = hyperparameters["env_id"]

        metadata = {}
        metadata["tags"] = [
              env_name,
              "reinforce",
              "reinforcement-learning",
              "custom-implementation",
              "deep-rl-class"
          ]

        # Add metrics
        eval = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
          )

        # Merges both dictionaries
        metadata = {**metadata, **eval}

        model_card = f"""
        # **Reinforce** Agent playing **{env_id}**
        This is a trained model of a **Reinforce** agent playing **{env_id}** .
        To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
        """

        readme_path = local_directory / "README.md"
        readme = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf8") as f:
              readme = f.read()
        else:
          readme = model_card

        with readme_path.open("w", encoding="utf-8") as f:
          f.write(readme)

        # Save our metrics to Readme metadata
        metadata_save(readme_path, metadata)

        # Step 6: Record a video
        video_path = local_directory / "replay.mp4"
        record_video(env, model, video_path, video_fps)

        # Step 7. Push everything to the Hub
        api.upload_folder(
              repo_id=repo_id,
              folder_path=local_directory,
              path_in_repo="",
        )

        print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Pixelcopter(PyGameWrapper):
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    """

    def __init__(self, width=48, height=48):
        actions = {
            "up": K_w
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.is_climbing = False
        self.speed = 0.0004 * width

    def _handle_player_events(self):
        self.is_climbing = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions['up']:
                    self.is_climbing = True

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * player velocity.
            * player distance to floor.
            * player distance to ceiling.
            * next block x distance to player.
            * next blocks top y location,
            * next blocks bottom y location.

            See code for structure.

        """

        min_dist = 999
        min_block = None
        for b in self.block_group:  # Groups do not return in order
            dist_to = b.pos.x - self.player.pos.x
            if dist_to > 0 and dist_to < min_dist:
                min_block = b
                min_dist = dist_to

        current_terrain = pygame.sprite.spritecollide(
            self.player, self.terrain_group, False)[0]
        state = {
            "player_y": self.player.pos.y,
            "player_vel": self.player.momentum,
            "player_dist_to_ceil": self.player.pos.y - (current_terrain.pos.y - self.height * 0.25),
            "player_dist_to_floor": (current_terrain.pos.y + self.height * 0.25) - self.player.pos.y,
            "next_gate_dist_to_player": min_dist,
            "next_gate_block_top": min_block.pos.y,
            "next_gate_block_bottom": min_block.pos.y + min_block.height
        }

        return self.getScreenRGB()

    def getScreenDims(self):
        return self.screen_dim

    def getActions(self):
        return self.actions.values()

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives <= 0.0

    def init(self):
        self.score = 0.0
        self.lives = 1.0

        self.player = HelicopterPlayer(
            self.speed,
            self.width,
            self.height
        )

        self.player_group = pygame.sprite.Group()
        self.player_group.add(self.player)

        self.block_group = pygame.sprite.Group()
        self._add_blocks()

        self.terrain_group = pygame.sprite.Group()
        self._add_terrain(0, self.width * 4)

    def _add_terrain(self, start, end):
        w = int(self.width * 0.1)
        # each block takes up 10 units.
        steps = range(start + int(w / 2), end + int(w / 2), w)
        y_jitter = []

        freq = 4.5 / self.width + self.rng.uniform(-0.01, 0.01)
        for step in steps:
            jitter = (self.height * 0.125) * \
                math.sin(freq * step + self.rng.uniform(0.0, 0.5))
            y_jitter.append(jitter)

        y_pos = [int((self.height / 2.0) + y_jit) for y_jit in y_jitter]

        for i in range(0, len(steps)):
            self.terrain_group.add(Terrain(
                (steps[i], y_pos[i]),
                self.speed,
                self.width,
                self.height
            )
            )

    def _add_blocks(self):
        x_pos = self.rng.randint(self.width, int(self.width * 1.5))
        y_pos = self.rng.randint(
            int(self.height * 0.25),
            int(self.height * 0.75)
        )
        self.block_group.add(
            Block(
                (x_pos, y_pos),
                self.speed,
                self.width,
                self.height
            )
        )

    def reset(self):
        self.init()

    def step(self, dt):

        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        self.player.update(self.is_climbing, dt)
        self.block_group.update(dt)
        self.terrain_group.update(dt)

        hits = pygame.sprite.spritecollide(
            self.player, self.block_group, False)
        for creep in hits:
            self.lives -= 1

        hits = pygame.sprite.spritecollide(
            self.player, self.terrain_group, False)
        for t in hits:
            if self.player.pos.y - self.player.height <= t.pos.y - self.height * 0.25:
                self.lives -= 1

            if self.player.pos.y >= t.pos.y + self.height * 0.25:
                self.lives -= 1

        for b in self.block_group:
            if b.pos.x <= self.player.pos.x and len(self.block_group) == 1:
                self.score += self.rewards["positive"]
                self._add_blocks()

            if b.pos.x <= -b.width:
                b.kill()

        for t in self.terrain_group:
            if t.pos.x <= -t.width:
                self.score += self.rewards["positive"]
                t.kill()

        if self.player.pos.y < self.height * 0.125:  # its above
            self.lives -= 1

        if self.player.pos.y > self.height * 0.875:  # its below the lowest possible block
            self.lives -= 1

        if len(self.terrain_group) <= (
                10 + 3):  # 10% per terrain, offset of ~2 with 1 extra
            self._add_terrain(self.width, self.width * 5)

        if self.lives <= 0.0:
            self.score += self.rewards["loss"]

        self.player_group.draw(self.screen)
        self.block_group.draw(self.screen)
        self.terrain_group.draw(self.screen)


class PixelcopterEnv(gym.Env):
    metadata = {"render_modes": ["human", "array", "rgb_array"]}

    def __init__(self, normalize=False, display=False, **kwargs):
        self.game_name = 'Game Name'
        self.init(normalize, display, **kwargs)

    def init(self, normalize, display, **kwargs):
        self.game = Pixelcopter(**kwargs)

        if display == False:
            # Do not open a PyGame window
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        if normalize:
            self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob_normalize, display_screen=display)
        else:
            self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob, display_screen=display)

        self.viewer = None
        self.action_set = self.gameOb.getActionSet()
        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.game.getGameState()),), dtype=np.float32)
        self.gameOb.init()

    def get_ob(self, state):
        return state

    def get_ob_normalize(self, state):
        raise NotImplementedError('Get observation normalize function is not implemented!')

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.gameOb.rng.seed(seed)
            self.gameOb.init()
        self.gameOb.reset_game()
        return self.gameOb.getGameState(), {}

    def step(self, action):
        reward = self.gameOb.act(self.action_set[action])
        terminated = self.gameOb.game_over()
        return self.gameOb.getGameState(), reward, terminated, False, {}

    def render(self, mode='human'):
        # img = self.gameOb.getScreenRGB()
        # img = self.gameOb.getScreenGrayscale()
        img = np.fliplr(np.rot90(self.gameOb.getScreenRGB(), 3))
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym_pygame.envs import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        if self.viewer != None:
            self.viewer.close()
            self.viewer = None
        return 0


env_id = 'Custom-Pixelcopter-ScreenRGB'
env = PixelcopterEnv()
eval_env = PixelcopterEnv()
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t

        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...


        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t]) # TODO: complete here

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()

        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}, Score Std: {:.2f}, Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque), np.std(scores_deque), np.mean(scores_deque) - np.std(scores_deque)
            ))

    return scores


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, h_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(h_size, h_size*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        s_size = s_size // 4
        self.linear = nn.Sequential(
            nn.Linear(h_size*2 * s_size * s_size, 128),
            nn.ReLU(),
            nn.Linear(128, a_size)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 255
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


pixelcopter_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 75000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
# torch.manual_seed(50)
pixelcopter_policy = Policy(pixelcopter_hyperparameters["state_space"], pixelcopter_hyperparameters["action_space"], pixelcopter_hyperparameters["h_size"]).to(device)
pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

scores = reinforce(pixelcopter_policy,
                   pixelcopter_optimizer,
                   pixelcopter_hyperparameters["n_training_episodes"],
                   pixelcopter_hyperparameters["max_t"],
                   pixelcopter_hyperparameters["gamma"],
                   1000)

repo_id = "nvasko/Reinforce-pixelcopter-cnn-1"
push_to_hub(
    repo_id,
    pixelcopter_policy,  # The model we want to save
    pixelcopter_hyperparameters,  # Hyperparameters
    eval_env,  # Evaluation environment
    video_fps=30
)
