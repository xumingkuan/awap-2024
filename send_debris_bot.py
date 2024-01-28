from src.player import Player
from src.map import Map
from src.robot_controller import RobotController
from src.game_constants import TowerType, Team, Tile, GameConstants, SnipePriority, get_debris_schedule
from src.debris import Debris
from src.tower import Tower
import random


class BotPlayer(Player):
    def __init__(self, map: Map):
        pass

    def play_turn(self, rc: RobotController):
        if rc.get_turn() > 600:
            if rc.can_send_debris(5, 101):
                rc.send_debris(5, 101)
            # if random.randint(0, 10) == 0:
            #     if rc.can_send_debris(7, 126):
            #         rc.send_debris(7, 126)
            # else:
            #     if rc.can_send_debris(2, 76):
            #         rc.send_debris(2, 76)
            # print(rc.get_balance(rc.get_ally_team()))
