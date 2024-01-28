import random
import numpy as np
from src.game_constants import SnipePriority, TowerType, Tile
from src.robot_controller import RobotController
from src.player import Player
from src.map import Map


class BotPlayer(Player):
    def __init__(self, map: Map):
        self.map = map
        self.path_len = 0.0
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
        self.grid_to_path = {}
        self.bomber_pattern = []
        self.reinforcer_pattern = []
        self.reinforcer_coverage = None
        self.best_reinforcer_loc = None
        self.tower_grid = None
        self.init_grid_coverage()
        self.best_solar_locations = []
        self.next_target_tower = TowerType.BOMBER
        self.is_beginning_2_gunships_1_bomber = False
        self.is_beginning_3_gunships_1_bomber = False
        self.is_beginning_4_gunships_1_bomber = False
        self.max_cd_to_compute = min(100, 10000 // len(self.map.path) + 1)
        self.guaranteed_bomber_damage = [[0 for _ in range(len(self.map.path))] for _ in range(self.max_cd_to_compute + 1)]
        self.enemy_has_solar_farm_ = False
        self.last_debris = None  # (cooldown, health, turn)
        self.send_debirs_interval = 200
        self.start_send_debris = False
        self.threat = 0
        self.opponent_max_towers = 0
        self.sold_solar_turn = -1

    def is_placeable(self, x, y):
        return self.map.is_space(int(x), int(y)) and self.tower_grid[x][y] is None

    def init_grid_coverage(self):
        self.path_len = len(self.map.path)
        path_grid = np.zeros(shape=(self.map.width, self.map.height), dtype=float)
        self.tower_grid = [[None for _ in range(self.map.height)] for _ in range(self.map.width)]

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

        path_extra_value = 0.1 / self.path_len ** 2  # we want to use bomber to kill debris earlier in the path, and use gunship to kill them later
        remaining_len = self.path_len - 1
        for t in self.map.path:
            path_grid[t[0], t[1]] = 1.0 + path_extra_value * remaining_len
            remaining_len -= 1
        path_grid_colsum = np.cumsum(path_grid, axis=1)

        self.bomber_coverage = np.array([[compute_bomber_coverage(x, y, path_grid_colsum) for y in range(self.map.height)] for x in range(self.map.width)])
        self.bomber_locations = np.dstack(np.unravel_index(np.argsort(-self.bomber_coverage.ravel()), self.bomber_coverage.shape))[0]
        self.bomber_locations = self.bomber_locations[
            np.array([self.map.tiles[self.bomber_locations[i][0]][self.bomber_locations[i][1]] == Tile.SPACE for i in range(len(self.bomber_locations))])]

        remaining_len = 0
        for t in self.map.path:
            path_grid[t[0], t[1]] = 1.0 + path_extra_value * remaining_len
            self.grid_to_path[(t[0], t[1])] = remaining_len
            remaining_len += 1
        path_grid_colsum = np.cumsum(path_grid, axis=1)

        self.gunship_coverage = np.array([[compute_gunship_coverage(x, y, path_grid_colsum) for y in range(self.map.height)] for x in range(self.map.width)])
        self.gunship_locations = np.dstack(np.unravel_index(np.argsort(-self.gunship_coverage.ravel()), self.gunship_coverage.shape))[0]
        self.gunship_locations = self.gunship_locations[
            np.array([self.map.tiles[self.gunship_locations[i][0]][self.gunship_locations[i][1]] == Tile.SPACE for i in range(len(self.gunship_locations))])]

        for i in range(-3, 4):
            for j in range(-3, 4):
                if i * i + j * j <= 10:
                    if not (i == 0 and j == 0):
                        self.bomber_pattern.append((i, j))

        for i in range(-2, 3):
            for j in range(-2, 3):
                if i * i + j * j <= 5:
                    if not (i == 0 and j == 0):
                        self.reinforcer_pattern.append((i, j))

    def compute_best_solar(self, rc: RobotController):
        ally_team = rc.get_ally_team()

        def compute_solar_coverage(x, y, pattern, rc: RobotController):
            if not self.is_placeable(int(x + pattern[-1][0]), int(y + pattern[-1][1])):
                return -1e10
            if not self.is_placeable(int(x + pattern[-2][0]), int(y + pattern[-2][1])):
                return -1e10
            ret = 0.0
            for p in pattern:
                if self.is_placeable(int(x + p[0]), int(y + p[1])):
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
        result_value = max(self.solar_coverage_1[ind1], self.solar_coverage_2[ind2])
        if result_value < 900:
            # do not build solar any more
            self.best_solar_locations = []

    def compute_best_reinforcer(self):
        self.reinforcer_coverage = np.zeros(shape=(self.map.width, self.map.height), dtype=float)
        for i in range(self.map.width):
            for j in range(self.map.height):
                self.reinforcer_coverage[i, j] = -4000 - self.bomber_coverage[i, j] ** 2 - self.gunship_coverage[i, j] ** 1.5
        for b in self.bombers:
            current_value = 1000 + self.bomber_coverage[b[0], b[1]] ** 2
            self.reinforcer_coverage[b[0], b[1]] -= current_value / 10.0
            for t in self.reinforcer_pattern:
                if self.map.is_space(int(b[0] + t[0]), int(b[1] + t[1])):
                    self.reinforcer_coverage[b[0] + t[0], b[1] + t[1]] += current_value
        for b in self.gunships:
            current_value = 1000 + self.gunship_coverage[b[0], b[1]] ** 1.5
            self.reinforcer_coverage[b[0], b[1]] -= current_value / 10.0
            for t in self.reinforcer_pattern:
                if self.map.is_space(int(b[0] + t[0]), int(b[1] + t[1])):
                    self.reinforcer_coverage[b[0] + t[0], b[1] + t[1]] += current_value
        for b in self.reinforcers:
            self.reinforcer_coverage[b[0], b[1]] -= 1e8  # already reinforcer
            for t in self.reinforcer_pattern:
                if self.map.is_space(int(b[0] + t[0]), int(b[1] + t[1])):
                    self.reinforcer_coverage[b[0] + t[0], b[1] + t[1]] -= 700  # diminishing return
        self.best_reinforcer_loc = np.unravel_index(np.argmax(self.reinforcer_coverage, axis=None), self.reinforcer_coverage.shape)

    def play_turn(self, rc: RobotController):
        if rc.get_turn() == 1:
            self.compute_best_solar(rc)
        current_opponent_towers = len(rc.get_towers(rc.get_enemy_team()))
        self.opponent_max_towers = max(self.opponent_max_towers, current_opponent_towers)
        if rc.get_turn() <= 550 + 14 * int(self.path_len) and current_opponent_towers <= self.opponent_max_towers - 2:
            self.sell_solars(rc)
            self.sold_solar_turn = rc.get_turn()
        if rc.get_turn() % 9 == 0 or (self.sold_solar_turn != -1 and rc.get_turn() - self.sold_solar_turn <= 100):
            if rc.get_turn() <= 1200 and not self.is_beginning_2_gunships_1_bomber:
                if self.beginning_2_gunships_1_bomber(rc):
                    self.is_beginning_2_gunships_1_bomber = True
            if rc.get_turn() <= 1200 and not self.is_beginning_3_gunships_1_bomber:
                if self.beginning_3_gunships_1_bomber(rc):
                    self.is_beginning_3_gunships_1_bomber = True
            if rc.get_turn() <= 2000 and not self.is_beginning_4_gunships_1_bomber:
                if self.beginning_4_gunships_1_bomber(rc):
                    self.is_beginning_4_gunships_1_bomber = True
        if rc.get_turn() >= 4000 and (rc.get_turn() % ((3000 // (len(self.solars) + 1)) + 1) == 0 or (
                rc.get_balance(rc.get_ally_team()) >= 5000 and rc.get_turn() % ((100 // (len(self.solars) + 1)) + 1) == 0)):
            self.sell_worst_solar(rc)
            if len(self.solars) < 10 and rc.get_balance(rc.get_ally_team()) >= 3000 and rc.get_turn() % 7 == 0:
                self.adjust_reinforcer(rc)
        self.compute_next_target_tower(rc)
        self.build_towers(rc)
        self.towers_attack(rc)
        self.send_debris(rc)

    def sell_worst_solar(self, rc: RobotController):
        if len(self.solars) == 0:
            return False
        solar_id = 0
        for i in range(1, len(self.solars)):
            if self.gunship_coverage[self.solars[i][0], self.solars[i][1]] > self.gunship_coverage[self.solars[solar_id][0], self.solars[solar_id][1]]:
                solar_id = i
        self.sell_solar(rc, self.solars[solar_id][0], self.solars[solar_id][1])

    def adjust_reinforcer(self, rc: RobotController):
        self.compute_best_reinforcer()
        for b in self.reinforcers:
            if self.reinforcer_coverage[b[0], b[1]] + 1e8 + 2000 < self.gunship_coverage[b[0], b[1]] ** 1.5:
                x = int(b[0])
                y = int(b[1])
                self.reinforcers.remove((x, y))
                t = self.find_tower(rc, x, y)
                rc.sell_tower(t.id)
                self.tower_grid[x][y] = None
                self.insert_to_gunship(x, y)
                self.insert_to_bomber(x, y)
                return True
        return False

    def update_guaranteed_bomber_damage(self, x, y, multiplier=1):
        # O(|self.max_cd_to_compute| * |len(self.map.path)|)
        path_locs = []
        for t in self.bomber_pattern:
            if (x + t[0], y + t[1]) in self.grid_to_path.keys():
                path_locs.append(self.grid_to_path[(x + t[0], y + t[1])])
        path_locs = sorted(path_locs)
        for i in range(1, self.max_cd_to_compute + 1):
            ptr = len(path_locs) - 1
            current_damage = 0
            current_duration = 0
            for j in reversed(range(len(self.map.path))):
                self.guaranteed_bomber_damage[i][j] += (current_damage + current_duration // TowerType.BOMBER.cooldown * TowerType.BOMBER.damage) * multiplier
                if ptr >= 0 and j == path_locs[ptr]:
                    current_duration += i
                    ptr -= 1
                else:  # not consecutive
                    current_damage += current_duration // TowerType.BOMBER.cooldown * TowerType.BOMBER.damage
                    current_duration = 0

    def build_bomber(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.BOMBER, x, y):
            rc.build_tower(TowerType.BOMBER, x, y)
            self.bombers.append((x, y))
            self.tower_grid[x][y] = TowerType.BOMBER
            self.update_guaranteed_bomber_damage(x, y)
            return True
        return False

    def build_gunship(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.GUNSHIP, x, y):
            rc.build_tower(TowerType.GUNSHIP, x, y)
            self.gunships.append((x, y))
            self.tower_grid[x][y] = TowerType.GUNSHIP
            return True
        return False

    def build_solar(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.SOLAR_FARM, x, y):
            rc.build_tower(TowerType.SOLAR_FARM, x, y)
            self.solars.append((x, y))
            self.tower_grid[x][y] = TowerType.SOLAR_FARM
            return True
        return False

    def insert_to_gunship(self, x, y):
        if len(self.gunship_locations) == 0:
            self.gunship_locations = np.array([(x, y)])
            return
        index = len(self.gunship_locations)
        for i in range(len(self.gunship_locations)):
            if self.gunship_coverage[self.gunship_locations[i][0], self.gunship_locations[i][1]] < self.gunship_coverage[x, y]:
                index = i
                break
        if index == len(self.gunship_locations):
            self.gunship_locations = np.concatenate((self.gunship_locations, [(x, y)]))
        else:
            self.gunship_locations = np.concatenate((self.gunship_locations[:index], [(x, y)], self.gunship_locations[index:]))

    def insert_to_bomber(self, x, y):
        if len(self.bomber_locations) == 0:
            self.bomber_locations = np.array([(x, y)])
            return
        index = len(self.bomber_locations)
        for i in range(len(self.bomber_locations)):
            if self.bomber_coverage[self.bomber_locations[i][0], self.bomber_locations[i][1]] < self.bomber_coverage[x, y]:
                index = i
                break
        if index == len(self.bomber_locations):
            self.bomber_locations = np.concatenate((self.bomber_locations, [(x, y)]))
        else:
            self.bomber_locations = np.concatenate((self.bomber_locations[:index], [(x, y)], self.bomber_locations[index:]))

    def sell_solar(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        self.solars.remove((x, y))
        t = self.find_tower(rc, x, y)
        rc.sell_tower(t.id)
        self.tower_grid[x][y] = None
        self.insert_to_gunship(x, y)
        self.insert_to_bomber(x, y)

    def sell_solars(self, rc):
        for s in self.solars:
            t = self.find_tower(rc, s[0], s[1])
            rc.sell_tower(t.id)
            self.tower_grid[s[0]][s[1]] = None
            self.insert_to_gunship(s[0], s[1])
            self.insert_to_bomber(s[0], s[1])
        self.solars.clear()
        self.compute_best_solar(rc)

    def find_tower(self, rc: RobotController, x, y):
        return rc.sense_towers_within_radius_squared(rc.get_ally_team(), x, y, 0)[0]

    def build_reinforcer(self, rc: RobotController, x, y):
        x = int(x)
        y = int(y)
        if rc.can_build_tower(TowerType.REINFORCER, x, y):
            rc.build_tower(TowerType.REINFORCER, x, y)
            self.reinforcers.append((x, y))
            self.tower_grid[x][y] = TowerType.REINFORCER
            return True
        elif rc.get_balance(rc.get_ally_team()) >= 1400 and self.tower_grid[x][y] is not None:
            if self.tower_grid[x][y] == TowerType.GUNSHIP and rc.get_balance(rc.get_ally_team()) >= 2200:
                self.gunships.remove((x, y))
                t = self.find_tower(rc, x, y)
                rc.sell_tower(t.id)
                rc.build_tower(TowerType.REINFORCER, x, y)
                self.reinforcers.append((x, y))
                self.tower_grid[x][y] = TowerType.REINFORCER
                return True
            elif self.tower_grid[x][y] == TowerType.BOMBER and rc.get_balance(rc.get_ally_team()) >= 1600:
                self.bombers.remove((x, y))
                t = self.find_tower(rc, x, y)
                rc.sell_tower(t.id)
                self.update_guaranteed_bomber_damage(x, y, multiplier=-1)  # cancel the update
                rc.build_tower(TowerType.REINFORCER, x, y)
                self.reinforcers.append((x, y))
                self.tower_grid[x][y] = TowerType.REINFORCER
                return True
            elif self.tower_grid[x][y] == TowerType.SOLAR_FARM and rc.get_balance(rc.get_ally_team()) >= 1400:
                self.solars.remove((x, y))
                t = self.find_tower(rc, x, y)
                rc.sell_tower(t.id)
                rc.build_tower(TowerType.REINFORCER, x, y)
                self.reinforcers.append((x, y))
                self.tower_grid[x][y] = TowerType.REINFORCER
                return True
        return False

    def greedy_build_bomber(self, rc: RobotController):
        ally_team = rc.get_ally_team()
        while len(self.bomber_locations) > 0 and not self.is_placeable(int(self.bomber_locations[0][0]), int(self.bomber_locations[0][1])):
            self.bomber_locations = self.bomber_locations[1:]
        if len(self.bomber_locations) > 0:
            if self.build_bomber(rc, self.bomber_locations[0][0], self.bomber_locations[0][1]):
                return True
        return False

    def greedy_build_gunship(self, rc: RobotController):
        ally_team = rc.get_ally_team()
        while len(self.gunship_locations) > 0 and not self.is_placeable(int(self.gunship_locations[0][0]), int(self.gunship_locations[0][1])):
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

    def beginning_2_gunships_1_bomber(self, rc: RobotController):
        debris = rc.get_debris(rc.get_ally_team())
        cnt = 0
        for d in debris:
            if TowerType.GUNSHIP.damage + TowerType.BOMBER.damage < d.health <= TowerType.GUNSHIP.damage * 2 + TowerType.BOMBER.damage:  # 2 gunships, 1 bomber
                cnt += 1
        if cnt >= 3:
            return True
        return False

    def beginning_3_gunships_1_bomber(self, rc: RobotController):
        debris = rc.get_debris(rc.get_ally_team())
        cnt = 0
        for d in debris:
            if TowerType.GUNSHIP.damage * 2 + TowerType.BOMBER.damage < d.health <= TowerType.GUNSHIP.damage * 3 + TowerType.BOMBER.damage:  # 3 gunships, 1 bomber
                cnt += 1
        if cnt >= 3:
            return True
        return False

    def beginning_4_gunships_1_bomber(self, rc: RobotController):
        debris = rc.get_debris(rc.get_ally_team())
        cnt = 0
        for d in debris:
            if TowerType.GUNSHIP.damage * 3 + TowerType.BOMBER.damage < d.health <= TowerType.GUNSHIP.damage * 4 + TowerType.BOMBER.damage:  # 3 gunships, 1 bomber
                cnt += 1
        if cnt >= 3:
            return True
        return False

    def safe_to_build_solar_first(self, rc: RobotController):
        while len(self.bomber_locations) > 0 and not self.is_placeable(int(self.bomber_locations[0][0]), int(self.bomber_locations[0][1])):
            self.bomber_locations = self.bomber_locations[1:]
        if len(self.bomber_locations) == 0:
            return False
        return False  # TODO

    def compute_next_target_tower(self, rc: RobotController):
        turns = rc.get_turn()
        if turns <= 3000 and len(self.bombers) == 0:
            self.next_target_tower = TowerType.BOMBER
        elif turns <= 1200 and self.is_beginning_2_gunships_1_bomber and len(self.gunships) < 2:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 1200 and self.is_beginning_3_gunships_1_bomber and len(self.gunships) < 3:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 2000 and self.is_beginning_4_gunships_1_bomber and len(self.gunships) < 4:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 3000 and len(rc.get_towers(rc.get_enemy_team())) == 0:
            self.next_target_tower = TowerType.REINFORCER  # auto bomber or gunship
        elif self.sold_solar_turn != -1 and turns - self.sold_solar_turn <= 100:
            self.next_target_tower = TowerType.REINFORCER  # auto bomber or gunship
        elif turns <= 600:  # 800
            self.next_target_tower = TowerType.SOLAR_FARM
        elif turns <= 3000 and len(self.gunships) == 0:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 3000 and self.threat / len(self.gunships) > 100000:
            if self.threat / len(self.gunships) > 1000000:
                self.sell_worst_solar(rc)
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 1200 and len(self.best_solar_locations) > 0:  # 1500
            self.next_target_tower = TowerType.SOLAR_FARM
        elif turns <= 3000 and len(self.bombers) < 2:
            self.next_target_tower = TowerType.BOMBER
        elif turns <= 3000 and len(self.gunships) < 3:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 2000 and len(self.best_solar_locations) > 0:  # 2400
            self.next_target_tower = TowerType.SOLAR_FARM
        elif turns <= 3000 and len(self.bombers) < 11:
            self.next_target_tower = TowerType.BOMBER
        elif turns <= 3030:  # 3200
            self.next_target_tower = TowerType.SOLAR_FARM
        elif turns <= 3450 and len(self.gunships) < 4:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 3450 and self.threat / len(self.gunships) > 50000:
            self.next_target_tower = TowerType.GUNSHIP
        elif turns <= 3450 and len(self.best_solar_locations) > 0 and len(self.solars) / self.map.width / self.map.height < 0.2:
            self.next_target_tower = TowerType.SOLAR_FARM
        else:
            self.next_target_tower = TowerType.REINFORCER

    def replace_with_reinforcers_or_build_gunship(self, rc: RobotController):
        turns = rc.get_turn()
        if len(rc.get_towers(rc.get_enemy_team())) == 0 or (self.sold_solar_turn != -1 and turns - self.sold_solar_turn <= 100):
            # auto bomber or gunship
            while len(self.bomber_locations) > 0 and not self.is_placeable(int(self.bomber_locations[0][0]), int(self.bomber_locations[0][1])):
                self.bomber_locations = self.bomber_locations[1:]
            if len(self.bomber_locations) > 0 and self.bomber_coverage[self.bomber_locations[0][0], self.bomber_locations[0][1]] >= 10:
                return self.greedy_build_bomber(rc)  # build bomber if it covers at least 10 tiles
            else:
                return self.greedy_build_gunship(rc)
        # endgame
        if self.best_reinforcer_loc is None:
            self.compute_best_reinforcer()
        if self.reinforcer_coverage[self.best_reinforcer_loc] > 0:
            # build reinforcer
            if self.build_reinforcer(rc, self.best_reinforcer_loc[0], self.best_reinforcer_loc[1]):
                self.compute_best_reinforcer()
                return True
            else:
                return False
        else:
            return self.greedy_build_gunship(rc)

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
        if self.next_target_tower == TowerType.REINFORCER:
            if self.replace_with_reinforcers_or_build_gunship(rc):
                self.compute_next_target_tower(rc)
                self.build_towers(rc)
            return

    def towers_attack(self, rc: RobotController):
        towers = rc.get_towers(rc.get_ally_team())
        for tower in towers:
            if tower.type == TowerType.BOMBER:
                rc.auto_bomb(tower.id)
        if len(self.gunships) == 0:
            return
        self.threat = 0.0  # clear last turn
        debris = rc.get_debris(rc.get_ally_team())
        debris_to_snipe = {}
        for d in debris:
            health = d.health - self.guaranteed_bomber_damage[min(d.total_cooldown, self.max_cd_to_compute)][d.progress]
            if health > 0:  # cannot be bombed
                debris_to_snipe[d.id] = (d.progress, health, d.total_cooldown)
        for tower in towers:
            if tower.type == TowerType.GUNSHIP and tower.current_cooldown <= 0:
                snipe_now = None
                best_val = None
                for k, v in debris_to_snipe.items():
                    if rc.can_snipe(tower.id, k):
                        if snipe_now is None or v > best_val:
                            snipe_now = k
                            best_val = v
                if snipe_now is not None:
                    rc.snipe(tower.id, snipe_now)
                    if best_val[1] - TowerType.GUNSHIP.damage <= 0:
                        # remove this debris
                        debris_to_snipe.pop(snipe_now)
                    else:
                        debris_to_snipe[snipe_now] = (best_val[0], best_val[1] - TowerType.GUNSHIP.damage, best_val[2])
        for k, v in debris_to_snipe.items():
            self.threat += max(1, v[1] * 100 - (self.path_len - v[0] - 1) * v[2])

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

    def enemy_has_solar_farm(self, rc: RobotController):
        if self.enemy_has_solar_farm_:
            return True
        for tower in rc.get_towers(rc.get_enemy_team()):
            if tower.type == TowerType.SOLAR_FARM:
                self.enemy_has_solar_farm_ = True
                return True
        return False

    def send_debris(self, rc: RobotController):
        turn = rc.get_turn()
        cooldown = 5
        threshold = max(20000, self.map.path_length * 900)
        if turn >= 3700 and turn % 200 < 100:
            if rc.get_balance(rc.get_ally_team()) >= threshold:
                self.start_send_debris = True
            if not self.start_send_debris:
                return
            cost_limit = rc.get_balance(rc.get_ally_team()) / (100 - turn % 100)
            l, r = 100, 1000000
            while l + 1 < r:
                mid = (l + r) // 2
                if rc.get_debris_cost(cooldown, mid) > cost_limit:
                    r = mid - 1
                else:
                    l = mid
            health = l
            if rc.can_send_debris(cooldown, health):
                rc.send_debris(cooldown, health)
        return

        group_size = 2
        cooldown = 1
        health = rc.get_health(rc.get_enemy_team())
        if rc.get_balance(rc.get_ally_team()) >= group_size * rc.get_debris_cost(cooldown, health):
            self.start_send_debris = True
        if self.start_send_debris:
            if rc.can_send_debris(cooldown, health):
                rc.send_debris(cooldown, health)
        return

        if rc.can_send_debris(1, rc.get_health(rc.get_enemy_team())):
            rc.send_debris(1, rc.get_health(rc.get_enemy_team()))
        return
        if rc.get_turn() % self.send_debirs_interval < self.send_debirs_interval * 0.5:
            return
        if rc.get_turn() <= 269:
            return
        if self.last_debris != None and self.last_debris[2] >= rc.get_turn() - self.send_debirs_interval:
            if rc.can_send_debris(self.last_debris[0], self.last_debris[1]):
                rc.send_debris(self.last_debris[0], self.last_debris[1])
            return
        if not self.enemy_has_solar_farm(rc):
            return
        # (1, 51) costs 220
        # (2, 76) costs 241
        max_cooldown = 6
        cost_limit = 241 if rc.get_turn() <= 1000 else rc.get_balance(rc.get_ally_team()) // 10 if rc.get_turn() <= 5000 else rc.get_balance(rc.get_ally_team())
        health_cand = [50 * x + 1 for x in range(1, 50)]
        for health in reversed(sorted(health_cand)):
            if rc.get_debris_cost(max_cooldown, health) > cost_limit:
                continue
            if rc.get_balance(rc.get_ally_team()) < 200:  # minimum for a debris
                break
            for cooldown in range(1, max_cooldown + 1):
                if rc.can_send_debris(cooldown, health) \
                        and rc.get_debris_cost(cooldown, health) <= cost_limit \
                        and not self.can_defense_debris(rc, (cooldown, health)):
                    self.last_debris = (cooldown, health, rc.get_turn())
                    rc.send_debris(cooldown, health)
                    self.send_debirs_interval *= 1.2
                    return
