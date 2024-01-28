# AWAP 2024 Game AI

See game engine at https://github.com/acm-cmu/awap-engine-2024-public.

Usage: clone this repo inside the `bots/` folder of the repo above, and follow the directions there. ElaineShiFanClub.py in the bots/ folder in the repo above.

## Strategy

- Initiate the grid at the beginning so that some operations become O(1) later.
- Compute the best "sunflower" location (enough to place the pattern and far from the debris path) in O(20 * map_size) when needed. Only called once every building 14 Solar Farms and 2 Reinforcers.
- If opponent sells at least 2 towers, sell all Solar Farms to build Bombers covering at least 10 tiles, or Gunships if there are no such locations.
- Build 1 Bomber at the beginning, then build Gunships/Bombers/Solar Farms in a predefined schedule.
- If opponent is sending debris that needs 2/3/4 Gunships together with 1 Bomber to kill, build them instead of Solar Farms in the schedule.
- Compute a threat value when attacking debris, and sell the Solar Farm with the best location for a Gunship when the value is too high.
- When attacking debris, conservatively estimate the damage to be dealt to each debris by all Bombers. Only attack the remaining amount using Gunships.
- Build Solar Farms until the 3450th turn, and make sure that they do not occupy more than 20% of the map size.
- In late game, build Reinforcers with favorable locations (possibly replacing other towers), build Gunships when there are no good locations for Reinforcers, and gradually remove Solar Farms.
- After removing Solar Farms, also gradually remove Reinforcers inside these Solar Farms when they cannot reinforce many Gunships nearby. Replace them with Gunships.
- Send debris only in late game, together with the natural debris with cooldown=5.
