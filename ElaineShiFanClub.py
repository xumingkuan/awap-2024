import random
import numpy as np
from src.game_constants import SnipePriority, TowerType, Tile
from src.robot_controller import RobotController
from src.player import Player
from src.map import Map
from src.debris import Debris
from src.game_state import GameState


class BotPlayer(Player):
    def __init__(self, map: Map):
        self.map = map
        self.bomber_coverage = None
        self.bomber_locations = None
        self.gunship_coverage = None
        self.gunship_locations = None
        self.bombers = []
        self.gunships = []
        self.init_grid_coverage()

    def init_grid_coverage(self):
        path_len = len(self.map.path)
        path_extra_value = 0.1 / path_len ** 2  # we want to kill debris earlier in the path
        path_grid = np.zeros(shape=(self.map.width, self.map.height), dtype=float)
        remaining_len = path_len - 1
        for t in self.map.path:
            path_grid[t[0], t[1]] = 1.0 + path_extra_value * remaining_len
            remaining_len -= 1
        path_grid_colsum = np.cumsum(path_grid, axis=1)

        def compute_bomber_coverage(x, y):
            # sqrt(10)
            ret = 0.0
            if x - 3 >= 0:
                ret += path_grid_colsum[x - 3, min(y + 1, self.map.height - 1)] - (0 if y - 2 < 0 else path_grid_colsum[x - 3, y - 2])
            if x - 2 >= 0:
                ret += path_grid_colsum[x - 2, min(y + 2, self.map.height - 1)] - (0 if y - 3 < 0 else path_grid_colsum[x - 2, y - 3])
            if x - 1 >= 0:
                ret += path_grid_colsum[x - 1, min(y + 3, self.map.height - 1)] - (0 if y - 4 < 0 else path_grid_colsum[x - 1, y - 4])
            ret += path_grid_colsum[x, min(y + 3, self.map.height - 1)] - (0 if y - 4 < 0 else path_grid_colsum[x, y - 4])
            if x + 1 < self.map.width:
                ret += path_grid_colsum[x + 1, min(y + 3, self.map.height - 1)] - (0 if y - 4 < 0 else path_grid_colsum[x + 1, y - 4])
            if x + 2 < self.map.width:
                ret += path_grid_colsum[x + 2, min(y + 2, self.map.height - 1)] - (0 if y - 3 < 0 else path_grid_colsum[x + 2, y - 3])
            if x + 3 < self.map.width:
                ret += path_grid_colsum[x + 3, min(y + 1, self.map.height - 1)] - (0 if y - 2 < 0 else path_grid_colsum[x + 3, y - 2])
            return ret

        def compute_gunship_coverage(x, y):
            # sqrt(60)
            ret = 0.0
            for dx in range(-7, 8):
                if 0 <= x + dx < self.map.width:
                    dy = int(np.sqrt(60 - dx * dx))
                    ret += path_grid_colsum[x + dx, min(y + dy, self.map.height - 1)] - (0 if y - dy - 1 < 0 else path_grid_colsum[x + dx, y - dy - 1])
            return ret

        self.bomber_coverage = np.array([[compute_bomber_coverage(x, y) for y in range(self.map.height)] for x in range(self.map.width)])
        self.bomber_locations = np.dstack(np.unravel_index(np.argsort(-self.bomber_coverage.ravel()), self.bomber_coverage.shape))[0]
        self.bomber_locations = self.bomber_locations[
            np.array([self.map.tiles[self.bomber_locations[i][0]][self.bomber_locations[i][1]] == Tile.SPACE for i in range(len(self.bomber_locations))])]

        self.gunship_coverage = np.array([[compute_gunship_coverage(x, y) for y in range(self.map.height)] for x in range(self.map.width)])
        self.gunship_locations = np.dstack(np.unravel_index(np.argsort(-self.gunship_coverage.ravel()), self.gunship_coverage.shape))[0]
        self.gunship_locations = self.gunship_locations[
            np.array([self.map.tiles[self.gunship_locations[i][0]][self.gunship_locations[i][1]] == Tile.SPACE for i in range(len(self.gunship_locations))])]

    def play_turn(self, rc: RobotController):
        self.build_towers(rc)
        self.towers_attack(rc)
        self.send_debris(rc)

    def build_bomber(self, rc: RobotController, x, y):
        if rc.can_build_tower(TowerType.BOMBER, x, y):
            rc.build_tower(TowerType.BOMBER, x, y)
            self.bombers.append((x, y))
            return True
        return False

    def build_gunship(self, rc: RobotController, x, y):
        if rc.can_build_tower(TowerType.GUNSHIP, x, y):
            rc.build_tower(TowerType.GUNSHIP, x, y)
            self.gunships.append((x, y))
            return True
        return False

    def greedy_build_bomber(self, rc: RobotController):
        ally_team = rc.get_ally_team()
        while len(self.bomber_locations) > 0 and not rc.is_placeable(ally_team, self.bomber_locations[0][0], self.bomber_locations[0][1]):
            self.bomber_locations = self.bomber_locations[1:]
        if len(self.bomber_locations) > 0:
            if self.build_bomber(rc, self.bomber_locations[0][0], self.bomber_locations[0][1]):
                return True
        return False

    def greedy_build_gunship(self, rc: RobotController):
        ally_team = rc.get_ally_team()
        while len(self.gunship_locations) > 0 and not rc.is_placeable(ally_team, self.gunship_locations[0][0], self.gunship_locations[0][1]):
            self.gunship_locations = self.gunship_locations[1:]
        if len(self.gunship_locations) > 0:
            if self.build_gunship(rc, self.gunship_locations[0][0], self.gunship_locations[0][1]):
                return True
        return False

    def build_towers(self, rc: RobotController):
        if len(self.bombers) == 0:
            # first bomber
            self.greedy_build_bomber(rc)
            if len(self.bombers) == 0:
                # wait for the first bomber
                return
        if len(self.bombers) - len(self.gunships) < 2:
            self.greedy_build_bomber(rc)
        else:
            self.greedy_build_gunship(rc)

    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.GUNSHIP:
                rc.auto_snipe(tower.id, SnipePriority.FIRST)
            elif tower.type == TowerType.BOMBER:
                rc.auto_bomb(tower.id)
                
    def can_defense_debris(self, rc: RobotController, debris_config):
        cooldown, health = debris_config
        towers = [tower for tower in rc.get_towers(rc.get_enemy_team()) if tower.type == TowerType.BOMBER]
        current_cooldown, current_health = cooldown, health
        progress = 0
        for _ in range(self.map.path_length * cooldown):
            current_cooldown = max(0, current_cooldown - 1)
            for tower in towers:
                tower.current_cooldown = max(0, tower.current_cooldown - 1)
            if current_cooldown <= 0:
                current_cooldown = cooldown
                progress += 1
                if progress == self.map.path_length:
                    return False
            debris_x, debris_y = self.map.path[progress]
            for tower in towers:
                if tower.type == TowerType.BOMBER \
                    and tower.current_cooldown <= 0 \
                    and (tower.x - debris_x)**2 + (tower.y - debris_y)**2 <= TowerType.BOMBER.range:
                    current_health -= TowerType.BOMBER.damage
                    if current_health <= 0:
                        return True
                    tower.current_cooldown = TowerType.BOMBER.cooldown
        return True
        
    def send_debris(self, rc: RobotController):
        if rc.get_turn() % 100 < 50:
            return
        cost_limit = 50 if rc.get_turn() <= 1000 else rc.get_balance(rc.get_ally_team()) // 100
        health_cand = [25 * x + 1 for x in range(100)] + [24]
        for health in reversed(sorted(health_cand)):
            for cooldown in range(1, 16):
                if rc.can_send_debris(cooldown, health) \
                    and rc.get_debris_cost(cooldown, health) <= cost_limit \
                    and not self.can_defense_debris(rc, (cooldown, health)):
                    rc.send_debris(cooldown, health)
                    