"""Static tool whitelist. Never build tool names dynamically."""

from app.tools.food_tools import search_foods
from app.tools.guide_tools import search_travel_guides
from app.tools.knowledge_tools import retrieve_tourism_knowledge
from app.tools.recommendation_tools import recommend_spots
from app.tools.spot_tools import get_spot_detail, search_scenic_spots

TOOLS = [
    search_scenic_spots,
    get_spot_detail,
    search_travel_guides,
    search_foods,
    retrieve_tourism_knowledge,
    recommend_spots,
]

__all__ = ["TOOLS"]
