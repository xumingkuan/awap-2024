import random
from src.game_constants import SnipePriority, TowerType
from src.robot_controller import RobotController
from src.player import Player
from src.map import Map
from src.debris import Debris
from src.game_state import GameState


class BotPlayer(Player):
    def __init__(self, map: Map):
        self.map = map

    def play_turn(self, rc: RobotController):
        self.build_towers(rc)
        self.towers_attack(rc)
        self.send_debris(rc)

    def build_towers(self, rc: RobotController):
        x = random.randint(0, self.map.height - 1)
        y = random.randint(0, self.map.height - 1)
        tower = random.randint(1, 4)  # randomly select a tower
        if (rc.can_build_tower(TowerType.GUNSHIP, x, y) and
                rc.can_build_tower(TowerType.BOMBER, x, y) and
                rc.can_build_tower(TowerType.SOLAR_FARM, x, y) and
                rc.can_build_tower(TowerType.REINFORCER, x, y)
        ):
            if tower == 1:
                rc.build_tower(TowerType.BOMBER, x, y)
            elif tower == 2:
                rc.build_tower(TowerType.GUNSHIP, x, y)
            elif tower == 3:
                rc.build_tower(TowerType.SOLAR_FARM, x, y)
            elif tower == 4:
                rc.build_tower(TowerType.REINFORCER, x, y)

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
                    