from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower


class BotPlayer(Player):
    def __init__(self, map: Map):
        pass

    def play_turn(self, rc: RobotController):
        if rc.get_turn() > 600 and rc.can_send_debris(7, 126):
            # print(rc.get_balance(rc.get_ally_team()))
            rc.send_debris(7, 126)
            # print(rc.get_balance(rc.get_ally_team()))
