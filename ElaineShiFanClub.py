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
        self.solar_pattern_1 = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (1, 2), (2, 2)]
        self.solar_pattern_2 = [(y, x) for (x, y) in self.solar_pattern_1]
        self.solar_coverage_1 = None
        self.solar_coverage_2 = None
        self.solars = []
        self.reinforcers = []
        self.init_grid_coverage()
        self.best_solar_locations = []
        self.next_target_tower = TowerType.BOMBER

    def init_grid_coverage(self):
        path_len = len(self.map.path)
        path_grid = np.zeros(shape=(self.map.width, self.map.height), dtype=float)

        def compute_bomber_coverage(x, y, path_grid_colsum):
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

        def compute_gunship_coverage(x, y, path_grid_colsum):
            # sqrt(60)
            ret = 0.0
            for dx in range(-7, 8):
                if 0 <= x + dx < self.map.width:
                    dy = int(np.sqrt(60 - dx * dx))
                    ret += path_grid_colsum[x + dx, min(y + dy, self.map.height - 1)] - (0 if y - dy - 1 < 0 else path_grid_colsum[x + dx, y - dy - 1])
            return ret

        path_extra_value = 0.1 / path_len ** 2  # we want to use bomber to kill debris earlier in the path, and use gunship to kill them later
        remaining_len = path_len - 1
        for t in self.map.path:
            path_grid[t[0], t[1]] = 1.0 + path_extra_value * remaining_len
            remaining_len -= 1
        path_grid_colsum = np.cumsum(path_grid, axis=1)

        self.bomber_coverage = np.array([[compute_bomber_coverage(x, y, path_grid_colsum) for y in range(self.map.height)] for x in range(self.map.width)])
        self.bomber_locations = np.dstack(np.unravel_index(np.argsort(-self.bomber_coverage.ravel()), self.bomber_coverage.shape))[0]
        self.bomber_locations = self.bomber_locations[
            np.array([self.map.tiles[self.bomber_locations[i][0]][self.bomber_locations[i][1]] == Tile.SPACE for i in range(len(self.bomber_locations))])]

        path_extra_value = 1.8 / path_len ** 2
        remaining_len = 0
        for t in self.map.path:
            path_grid[t[0], t[1]] = 1.0 + path_extra_value * remaining_len
            remaining_len += 1
        path_grid_colsum = np.cumsum(path_grid, axis=1)

        self.gunship_coverage = np.array([[compute_gunship_coverage(x, y, path_grid_colsum) for y in range(self.map.height)] for x in range(self.map.width)])
        self.gunship_locations = np.dstack(np.unravel_index(np.argsort(-self.gunship_coverage.ravel()), self.gunship_coverage.shape))[0]
        self.gunship_locations = self.gunship_locations[
            np.array([self.map.tiles[self.gunship_locations[i][0]][self.gunship_locations[i][1]] == Tile.SPACE for i in range(len(self.gunship_locations))])]

    def compute_best_solar(self, rc: RobotController):
        ally_team = rc.get_ally_team()

        def compute_solar_coverage(x, y, pattern, rc: RobotController):
            if not rc.is_placeable(ally_team, int(x + pattern[-1][0]), int(y + pattern[-1][1])):
                return -1e10
            if not rc.is_placeable(ally_team, int(x + pattern[-2][0]), int(y + pattern[-2][1])):
                return -1e10
            ret = 0.0
            for p in pattern:
                if rc.is_placeable(ally_team, int(x + p[0]), int(y + p[1])):
                    ret += 1000 - self.bomber_coverage[x + p[0], y + p[1]] ** 2 - self.gunship_coverage[x + p[0], y + p[1]] ** 1.5
            return ret

        self.solar_coverage_1 = np.array(
            [[compute_solar_coverage(x, y, self.solar_pattern_1, rc) for y in range(self.map.height - 4)] for x in range(self.map.width - 3)])
        self.solar_coverage_2 = np.array(
            [[compute_solar_coverage(x, y, self.solar_pattern_2, rc) for y in range(self.map.height - 3)] for x in range(self.map.width - 4)])
        ind1 = np.unravel_index(np.argmax(self.solar_coverage_1, axis=None), self.solar_coverage_1.shape)
        ind2 = np.unravel_index(np.argmax(self.solar_coverage_2, axis=None), self.solar_coverage_2.shape)
        if self.solar_coverage_1[ind1] > self.solar_coverage_2[ind2]:
            self.best_solar_locations = [(x + ind1[0], y + ind1[1]) for (x, y) in self.solar_pattern_1]
        else:
            self.best_solar_locations = [(x + ind2[0], y + ind2[1]) for (x, y) in self.solar_pattern_2]
        if self.solar_coverage_1[ind1] < 900 and self.solar_coverage_2[ind2] < 900:
            # do not build solar any more
            self.best_solar_locations = []

    def play_turn(self, rc: RobotController):
        if rc.get_turn() == 1:
            self.compute_best_solar(rc)
            exit()
        self.send_debris(rc)
        self.compute_next_target_tower(rc)
        self.build_towers(rc)
        self.towers_attack(rc)

    def build_bomber(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.BOMBER, x, y):
            rc.build_tower(TowerType.BOMBER, x, y)
            self.bombers.append((x, y))
            return True
        return False

    def build_gunship(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.GUNSHIP, x, y):
            rc.build_tower(TowerType.GUNSHIP, x, y)
            self.gunships.append((x, y))
            return True
        return False

    def build_solar(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.SOLAR_FARM, x, y):
            rc.build_tower(TowerType.SOLAR_FARM, x, y)
            self.solars.append((x, y))
            return True
        return False

    def build_reinforcer(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.REINFORCER, x, y):
            rc.build_tower(TowerType.REINFORCER, x, y)
            self.reinforcers.append((x, y))
            return True
        return False

    def greedy_build_bomber(self, rc: RobotController):
        ally_team = rc.get_ally_team()
        while len(self.bomber_locations) > 0 and not rc.is_placeable(ally_team, int(self.bomber_locations[0][0]), int(self.bomber_locations[0][1])):
            self.bomber_locations = self.bomber_locations[1:]
        if len(self.bomber_locations) > 0:
            if self.build_bomber(rc, self.bomber_locations[0][0], self.bomber_locations[0][1]):
                return True
        return False

    def greedy_build_gunship(self, rc: RobotController):
        ally_team = rc.get_ally_team()
        while len(self.gunship_locations) > 0 and not rc.is_placeable(ally_team, int(self.gunship_locations[0][0]), int(self.gunship_locations[0][1])):
            self.gunship_locations = self.gunship_locations[1:]
        if len(self.gunship_locations) > 0:
            if self.build_gunship(rc, self.gunship_locations[0][0], self.gunship_locations[0][1]):
                return True
        return False

    def greedy_build_best_solar(self, rc: RobotController):
        if len(self.best_solar_locations) == 0:
            return False
        ret = False
        for (x, y) in self.best_solar_locations[:-2]:
            if self.build_solar(rc, x, y):
                ret = True
        for (x, y) in self.best_solar_locations[-2:]:
            if self.build_reinforcer(rc, x, y):
                ret = True
        if not ret and rc.get_balance(rc.get_ally_team()) >= TowerType.REINFORCER.cost:
            self.compute_best_solar(rc)
            if len(self.best_solar_locations) > 0:
                if self.greedy_build_best_solar(rc):  # keep building
                    ret = True
        return ret

    def compute_next_target_tower(self, rc: RobotController):
        turns = rc.get_turn()
        if len(self.bombers) == 0:
            self.next_target_tower = TowerType.BOMBER
        elif turns <= 600:  # 800
            self.next_target_tower = TowerType.SOLAR_FARM
        elif len(self.gunships) == 0:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 1200:  # 1500
            self.next_target_tower = TowerType.SOLAR_FARM
        elif len(self.bombers) < 2:
            self.next_target_tower = TowerType.BOMBER
        elif len(self.gunships) < 3:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 2000:  # 2400
            self.next_target_tower = TowerType.SOLAR_FARM
        elif len(self.bombers) < 11:
            self.next_target_tower = TowerType.BOMBER
        elif turns <= 3030:  # 3200
            self.next_target_tower = TowerType.SOLAR_FARM
        elif len(self.gunships) < 4:
            self.next_target_tower = TowerType.GUNSHIP
        else:
            self.next_target_tower = TowerType.GUNSHIP

    def build_towers(self, rc: RobotController):
        if self.next_target_tower == TowerType.BOMBER:
            if self.greedy_build_bomber(rc):
                self.compute_next_target_tower(rc)
                self.build_towers(rc)
            return  # if not built, wait for it
        if self.next_target_tower == TowerType.GUNSHIP:
            if self.greedy_build_gunship(rc):
                self.compute_next_target_tower(rc)
                self.build_towers(rc)
            return
        if self.next_target_tower == TowerType.SOLAR_FARM:
            if self.greedy_build_best_solar(rc):
                self.compute_next_target_tower(rc)
                self.build_towers(rc)
            return

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
                        and (tower.x - debris_x) ** 2 + (tower.y - debris_y) ** 2 <= TowerType.BOMBER.range:
                    current_health -= TowerType.BOMBER.damage
                    if current_health <= 0:
                        return True
                    tower.current_cooldown = TowerType.BOMBER.cooldown
        return True

    def send_debris(self, rc: RobotController):
        if rc.get_turn() % 249 < 200:
            return
        cost_limit = 50 if rc.get_turn() <= 1000 else rc.get_balance(rc.get_ally_team()) // 100
        health_cand = [25 * x + 1 for x in range(100)] + [24]
        for health in reversed(sorted(health_cand)):
            for cooldown in range(1, 16):
                if rc.can_send_debris(cooldown, health) \
                        and rc.get_debris_cost(cooldown, health) <= cost_limit \
                        and not self.can_defense_debris(rc, (cooldown, health)):
                    rc.send_debris(cooldown, health)
