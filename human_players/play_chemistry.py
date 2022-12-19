# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple human player for testing `chemistry`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to select the `endocytose` action.
Use `TAB` to switch between players.
"""

import argparse
import json
from ml_collections import config_dict

from meltingpot.python.configs.substrates import chemistry__three_metabolic_cycles
from meltingpot.python.configs.substrates import chemistry__three_metabolic_cycles_with_plentiful_distractors
from meltingpot.python.configs.substrates import chemistry__two_metabolic_cycles
from meltingpot.python.configs.substrates import chemistry__two_metabolic_cycles_with_distractors
from meltingpot.python.human_players import level_playing_utils


MAX_SCREEN_WIDTH = 800
MAX_SCREEN_HEIGHT = 600
FRAMES_PER_SECOND = 8


_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'ioAction': level_playing_utils.get_space_key_pressed,
}

environment_configs = {
    'chemistry__three_metabolic_cycles': (
        chemistry__three_metabolic_cycles),
    'chemistry__three_metabolic_cycles_with_plentiful_distractors': (
        chemistry__three_metabolic_cycles_with_plentiful_distractors),
    'chemistry__two_metabolic_cycles': chemistry__two_metabolic_cycles,
    'chemistry__two_metabolic_cycles_with_distractors': (
        chemistry__two_metabolic_cycles_with_distractors),
}


def verbose_fn(unused_env, unused_player_index):
  """Activate verbose printing with --verbose=True."""
  pass


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='chemistry__two_metabolic_cycles',
      help='Level name to load')
  parser.add_argument(
      '--observation', type=str, default='RGB', help='Observation to render')
  parser.add_argument(
      '--settings', type=json.loads, default={}, help='Settings as JSON string')
  # Activate verbose mode with --verbose=True.
  parser.add_argument(
      '--verbose', type=bool, default=False, help='Print debug information')
  # Activate events printing mode with --print_events=True.
  parser.add_argument(
      '--print_events', type=bool, default=False, help='Print events')

  args = parser.parse_args()
  env_module = environment_configs[args.level_name]
  env_config = env_module.get_config()
  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)
  level_playing_utils.run_episode(
      args.observation, args.settings, _ACTION_MAP,
      env_config, level_playing_utils.RenderType.PYGAME,
      verbose_fn=verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
  main()
