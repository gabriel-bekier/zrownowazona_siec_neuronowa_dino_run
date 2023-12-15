import pyautogui
import pydirectinput
import cv2 
import numpy as np
import pytesseract
import concurrent.futures
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
import torch
from torchviz import make_dot

TRAIN_DIR = './train/'
LOG_DIR = './LOG/'
NUM_THREADS = 4
NUM_OF_STEPS = 2000

class CustomCallback(BaseCallback):
  def __init__(self, env, check_freq, save_path, thread_id, verbose=1 ):
    super(CustomCallback, self).__init__(verbose)
    self.env = env
    self.thread_id = thread_id
    self.best_reward = float('-inf')
    self.check_freq = check_freq
    self.save_path = save_path

  def _init_callback(self):
    if self.save_path is not None:
      os.makedirs(self.save_path, exist_ok=True)

  def _on_step(self) -> bool:
    current_reward = self.env.best_reward
    if current_reward > self.best_reward:
        self.best_reward = current_reward

    self.log_best_reward()
    if self.n_calls % self.check_freq == 0:
      model_path = os.path.join(self.save_path, 'best_model_{}_{}'.format(self.n_calls, self.thread_id))
      self.model.save(model_path)
    return True

  def log_best_reward(self):
    log_file = os.path.join(LOG_DIR, 'best_rewards.txt')
    with open(log_file, 'a') as f:
        f.write(f"Timestep {self.num_timesteps}\n")
        f.write(f"Thread {self.env.thread_id}: {self.best_reward}\n")
        f.write("\n")

class WebGame(Env):
  def __init__(self, thread_id):
    super().__init__()
 
    self.observation_space = Box(low=0, high=255, shape=(1,100,312), dtype=np.uint8)
    self.action_space = Discrete(3)
    self.best_reward = float('-inf')
    self.current_reward = 0
    self.thread_id = thread_id
    self.click_location = [
      [425, 320],
      [1375, 320],
      [425, 870],
      [1375, 870]
    ][thread_id]
    
    self.game_location = {
        0: {'top': 200, 'left': 0, 'width': 750, 'height': 240},
        1: {'top': 200, 'left': 950, 'width': 750, 'height': 240},
        2: {'top': 750, 'left': 0, 'width': 750, 'height': 240},
        3 : {'top': 750, 'left': 950, 'width': 750, 'height': 240}
    }[thread_id]

    self.done_location = {
        0: {'top': 220, 'left': 240, 'width': 400, 'height': 60},
        1: {'top': 220, 'left': 1190, 'width': 400, 'height': 60},
        2: {'top': 760, 'left': 240, 'width': 400, 'height': 60},
        3: {'top': 760, 'left': 1190, 'width': 400, 'height': 60}
    }[thread_id]
  def step(self, action):
    action_map = {
      0:'space',
      1:'down',
      2:'no_op'
    }

    if action != 2:
      pydirectinput.click(x=self.click_location[0], y=self.click_location[1])
      pydirectinput.press(action_map[action])

    done, done_cap = self.get_done()
    observation = self.get_observation()

    reward = 1
    self.current_reward += reward
    terminated = done
    truncated = False
    info = {}

    if terminated:
      self.best_reward = max(self.best_reward, self.current_reward)
      self.current_reward = 0 

    return observation, reward, terminated, truncated, info

  def reset(self, seed=None):
    time.sleep(1)
    pydirectinput.click(x=self.click_location[0], y=self.click_location[1])
    pydirectinput.press('space')
    return self.get_observation(), {}
  
  def render(self):
    pass

  def close(self):
    cv2.destroyAllWindows()
  def get_observation(self):
    raw = pyautogui.screenshot(region=(self.game_location['left'],
                               self.game_location['top'],
                               self.game_location['width'],
                               self.game_location['height']))
    gray = cv2.cvtColor(np.array(raw), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (312, 100))

    channel = np.squeeze(np.reshape(resized, (1,100,312)))

    return channel
  
  def get_done(self):
    done_cap = pyautogui.screenshot(region=(self.done_location['left'],
                               self.done_location['top'],
                               self.done_location['width'],
                               self.done_location['height']))
    done = False
    text = pytesseract.image_to_string(np.array(done_cap)).strip()

    if (len(text) > 0):
      done = True

    return done, done_cap

def train_model(model_id, env):
  try:
    # TODO -> model cpu
    model = DQN('CnnPolicy', env, tensorboard_log=f'./logs_{model_id}/thread_{model_id}', verbose=1, buffer_size=15000, learning_starts=0)
    # TODO -> model gpu
    # model = DQN('CnnPolicy', env, device='cuda', tensorboard_log=f'./logs_{model_id}/thread_{model_id}', verbose=1, buffer_size=15000, learning_starts=0)
    callback = CustomCallback(env, check_freq=NUM_OF_STEPS, save_path=TRAIN_DIR, thread_id=model_id)
    model.learn(total_timesteps=NUM_OF_STEPS, callback=callback)
  except Exception as e:
      print(f"Exception in thread {model_id}: {e}")
  finally:
      env.close()

def visualize_saved_models(directory):
    models = []
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            model_path = os.path.join(directory, filename)
            model = DQN.load(model_path)
            models.append(model)

    for i, model in enumerate(models):
        q_network = model.policy.q_net
        # TODO -> model cpu
        input_obs = torch.tensor(model.observation_space.sample(), dtype=torch.float32).unsqueeze(0)
        # TODO -> model gpu
        # input_obs = torch.tensor(model.observation_space.sample(), dtype=torch.float32, device='cuda').unsqueeze(0)
        output = q_network(input_obs)
        dot = make_dot(output, params=dict(q_network.named_parameters()))
        dot.render(filename=f'neural_network_{i}', format='png')


def main():
  envs = [WebGame(thread_id=i) for i in range(NUM_THREADS)]

  with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(train_model, model_id, env) for model_id, env in enumerate(envs)]

    concurrent.futures.wait(futures)

  visualize_saved_models(TRAIN_DIR)  
  

if __name__ == "__main__":
  main()
