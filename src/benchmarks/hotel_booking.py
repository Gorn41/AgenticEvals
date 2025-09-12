"""
Goal-Based Hotel Booking Benchmark for AgenticEvals.

This benchmark tests a model's ability to infer search constraints from natural language
client requests and plan systematic searches to cover the required search space.

The model must:
1. Parse natural language client stories to identify hard constraints
2. Plan searches that cover the minimum required combinations  
3. Use only simple search interface: location, checkin_date, checkout_date, guests

No actual hotel searching is performed - only search space planning is evaluated.
"""

import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchStep:
    """A single search step with concrete parameters."""
    step_number: int
    location: str
    checkin: str
    checkout: str
    guests: int
    
    def __str__(self):
        return f"Search {self.step_number}: location={self.location}, dates={self.checkin} to {self.checkout}, guests={self.guests}"


@dataclass
class SearchPlan:
    """A sequence of search steps covering the required search space."""
    steps: List[SearchStep]
    
    def __post_init__(self):
        if not isinstance(self.steps, list):
            self.steps = []


@dataclass 
class HardConstraints:
    """Hard constraints extracted from natural language client story."""
    required_locations: List[str]
    required_dates: List[str]  # List of "YYYY-MM-DD to YYYY-MM-DD" strings
    guests: int
    location_date_pairs: Optional[List[tuple]] = None  # For fixed pairings: [(location, date_range), ...]
    
    def get_required_combinations(self) -> List[tuple]:
        """Get all required (location, date) combinations."""
        if self.location_date_pairs:
            # Fixed pairings: each location-date pair is a specific requirement
            return [(loc, date, self.guests) for loc, date in self.location_date_pairs]
        else:
            # Flexible cross-product: search all combinations
            combinations = []
            for location in self.required_locations:
                for date_range in self.required_dates:
                    combinations.append((location, date_range, self.guests))
            return combinations


@benchmark(
    name="hotel_booking",
    agent_type=AgentType.GOAL_BASED,
    description="Hotel booking benchmark with natural language constraint inference"
)
class HotelBookingBenchmark(BaseBenchmark):
    """Hotel booking benchmark focusing purely on search space planning."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
    
    def get_tasks(self) -> List[Task]:
        """Get clean hotel booking planning tasks."""
        tasks = []
        
        # Define simple scenarios with first-person client stories
        scenarios = [
            {
                "name": "Business Trip",
                "client_story": "I need a hotel for my business trip next month. I'm finalizing meetings with potential clients in either city_a or city_b for March 15-17. I won't know which city until they confirm, so I need to research accommodation options in both cities to be prepared. My colleague will be joining me for the trip.",
                "goal": "Find accommodation options in potential meeting cities",
                "hard_constraints": {
                    "required_locations": ["city_a", "city_b"],
                    "required_dates": ["2024-03-15 to 2024-03-17"],
                    "guests": 2
                },
                "expected_searches": 2,
                "difficulty": "simple"
            },
            
            {
                "name": "Family Visit", 
                "client_story": "I'm planning to visit family and need accommodation. My parents live in city_c and my sister lives in city_d. I can make the trip either March 10-12 or March 15-17, depending on when I can get time off work. There will be 4 of us traveling together.",
                "goal": "Find accommodation near family in either city during available dates",
                "hard_constraints": {
                    "required_locations": ["city_c", "city_d"],
                    "required_dates": ["2024-03-10 to 2024-03-12", "2024-03-15 to 2024-03-17"],
                    "guests": 4
                },
                "expected_searches": 4,
                "difficulty": "medium"
            },
            
            {
                "name": "Conference Attendance",
                "client_story": "I need to attend a conference that's being held simultaneously in three cities. The main sessions are in city_a, but there are also important workshops in city_e and city_f. I need to pick one location for my stay from March 20-22. I'll be traveling alone.",
                "goal": "Find accommodation in one of the conference cities",
                "hard_constraints": {
                    "required_locations": ["city_a", "city_e", "city_f"], 
                    "required_dates": ["2024-03-20 to 2024-03-22"],
                    "guests": 1
                },
                "expected_searches": 3,
                "difficulty": "simple"
            },
            
            {
                "name": "Wedding Weekend",
                "client_story": "I'm attending my friend's wedding and need accommodation. The wedding venues are in city_b and city_g. The rehearsal dinner is Friday April 5th, the wedding is Saturday April 6th, and there's a brunch on Sunday April 7th. I could stay April 5-7, April 6-8, or April 5-8 depending on which events I attend. My partner and I will be traveling together.",
                "goal": "Find accommodation near wedding venues during event dates", 
                "hard_constraints": {
                    "required_locations": ["city_b", "city_g"],
                    "required_dates": ["2024-04-05 to 2024-04-07", "2024-04-06 to 2024-04-08", "2024-04-05 to 2024-04-08"],
                    "guests": 2
                },
                "expected_searches": 6,
                "difficulty": "complex"
            },
            
            {
                "name": "Business Conference Circuit",
                "client_story": "I need to attend a series of tech conferences this May. The main event is in city_a from May 15-17, but there are satellite events in nearby cities. City_c and city_e host related workshops, and city_f has the startup showcase. I need to be strategic about where I stay since I might need to travel between cities. The main conference in city_a is May 15-17, the workshop in city_c is May 16-17, the workshop in city_e is May 18-19, and the startup showcase in city_f is May 19-20. I need to find accommodation that allows me to attend all these connected events. I'll be traveling solo.",
                "goal": "Find strategic accommodation for multi-city conference attendance",
                "hard_constraints": {
                    "location_date_pairs": [
                        ("city_a", "2024-05-15 to 2024-05-17"),  # Main conference
                        ("city_c", "2024-05-16 to 2024-05-17"),  # Workshop 1
                        ("city_e", "2024-05-18 to 2024-05-19"),  # Workshop 2  
                        ("city_f", "2024-05-19 to 2024-05-20")   # Startup showcase
                    ],
                    "required_locations": ["city_a", "city_c", "city_e", "city_f"],
                    "required_dates": ["2024-05-15 to 2024-05-17", "2024-05-16 to 2024-05-17", "2024-05-18 to 2024-05-19", "2024-05-19 to 2024-05-20"],
                    "guests": 1
                },
                "expected_searches": 4,  # Each conference has specific dates
                "difficulty": "very_complex"
            },
            
            {
                "name": "Vacation Planning",
                "client_story": "We're planning a vacation and considering three different cities. We're thinking about city_c for the museums, city_d for the beaches, or city_e for the mountains. We're flexible on dates - we could go May 10-12, May 15-17, or May 20-22. There will be 3 of us on this trip.",
                "goal": "Find accommodation in preferred vacation destinations during flexible dates",
                "hard_constraints": {
                    "required_locations": ["city_c", "city_d", "city_e"],
                    "required_dates": ["2024-05-10 to 2024-05-12", "2024-05-15 to 2024-05-17", "2024-05-20 to 2024-05-22"],
                    "guests": 3
                },
                "expected_searches": 9,
                "difficulty": "complex"
            },
            
            {
                "name": "Medical Conference with Distractors",
                "client_story": "I'm a doctor attending a medical conference. My hospital is sponsoring me, and they want me to consider multiple locations for strategic reasons. The conference is happening in city_a, city_c, and city_e. I initially thought about city_b for its restaurants and city_d for its nightlife, but those don't actually have the conference. My wife wanted to come along to city_f where her college friend lives, but that's not where the conference is either. I also considered city_g because of the shopping, but again, no conference there. The dates are March 25-27, and I'll be traveling alone for this professional event.",
                "goal": "Find accommodation in actual conference cities, ignoring non-conference distractors",
                "hard_constraints": {
                    "required_locations": ["city_a", "city_c", "city_e"],
                    "required_dates": ["2024-03-25 to 2024-03-27"],
                    "guests": 1
                },
                "expected_searches": 3,
                "difficulty": "hard_with_distractors"
            },
            
            {
                "name": "Multi-Generational Family Reunion with Distractors",
                "client_story": "We're organizing a family reunion and it's quite complex. Initially, we thought about city_h because that's where great-grandma was born, and city_i because Uncle Bob lives there now. However, the actual venues that can accommodate our large group are only in city_a and city_c. We also looked into city_j for the historical significance and city_k for the theme parks, but they don't have the right facilities. Some cousins suggested city_l for the beaches, but that's too far from the airport. The reunion committee has narrowed it down to the two cities with proper banquet halls. We're flexible between June 15-17 or June 22-24, depending on venue availability. There will be 6 of us in the core planning group who need accommodation.",
                "goal": "Find accommodation in cities with actual reunion venues, ignoring location distractors",
                "hard_constraints": {
                    "required_locations": ["city_a", "city_c"],
                    "required_dates": ["2024-06-15 to 2024-06-17", "2024-06-22 to 2024-06-24"],
                    "guests": 6
                },
                "expected_searches": 4,
                "difficulty": "hard_with_distractors"
            },
            
            {
                "name": "Corporate Training with Budget Distractors",
                "client_story": "My company is sending me to a mandatory training program. HR initially suggested city_m for cost savings and city_n for the 'team building environment', but those locations don't actually offer the specific certification we need. The training is only available in city_c, city_e, and city_f. I personally would prefer city_o for the weather or city_p for the cultural attractions, but work requirements come first. My manager mentioned city_q as an option, but that's for a different training program entirely. The certified training provider offers sessions in each of the three cities during multiple date ranges: April 10-12, April 15-17, and April 20-22. I need to attend one session, and I'm flexible on both location and timing since all three cities offer the training during all three date windows. I'll be traveling alone.",
                "goal": "Find accommodation in cities with actual required training, ignoring preference and budget distractors",
                "hard_constraints": {
                    "required_locations": ["city_c", "city_e", "city_f"],
                    "required_dates": ["2024-04-10 to 2024-04-12", "2024-04-15 to 2024-04-17", "2024-04-20 to 2024-04-22"],
                    "guests": 1
                },
                "expected_searches": 9,
                "difficulty": "hard_with_distractors"
            },
            
            {
                "name": "Wedding Circuit with Social Distractors",
                "client_story": "I have multiple weddings to attend this spring - it's wedding season! My college roommate is getting married in city_a and gave me two possible date options: May 5-7 or May 12-14, since they're flexible with their venue. My cousin's wedding is in city_d and they also offered flexibility - I could attend either May 5-7 or May 12-14 depending on my schedule. I was also invited to my coworker's wedding in city_r, but I realized I can't make that one due to scheduling conflicts. My high school friend's wedding in city_s sounds amazing, but it's the same weekend as one of my other options and I already committed to family. There's also a wedding in city_t that would be fun, but I don't know the couple well enough to justify the travel. I should also mention that my ex-boyfriend's wedding is in city_u, but obviously I won't be attending that one. Two weddings are definitely enough for one month, and I want to explore all the date combinations to find the best schedule. I'll be traveling with my partner for both weddings we're actually attending.",
                "goal": "Find accommodation for weddings with flexible date options, ignoring declined/conflict distractors",
                "hard_constraints": {
                    "required_locations": ["city_a", "city_d"],
                    "required_dates": ["2024-05-05 to 2024-05-07", "2024-05-12 to 2024-05-14"],
                    "guests": 2
                },
                "expected_searches": 4,
                "difficulty": "hard_with_distractors"
            },
            
            {
                "name": "Academic Conference Circuit with Extensive Distractors",
                "client_story": "I'm a university professor planning my conference season, and it's absolutely overwhelming this year. My research focuses on computational linguistics, but I'm also interested in cognitive science, machine learning, and natural language processing. The International Conference on Computational Linguistics is happening in city_a from June 10-13, which is perfect for my core research. However, I also want to attend the Workshop on Cognitive Modeling in city_e, which runs June 12-14 - there's some overlap, but I think I can manage both. The Association for Computational Linguistics meeting is in city_f from June 15-18, and that's absolutely crucial for my tenure review. There's also a symposium on Neural Information Processing in city_h from June 20-22, but honestly, I'm not sure if my department will approve the travel budget for that one. My colleague mentioned a fascinating workshop on Semantic Web Technologies in city_i, but that's in July and I'll be on vacation then. The International Joint Conference on Artificial Intelligence is in city_j, but registration is already closed. I was also considering the Workshop on Multimodal Learning in city_k, but my co-author can't make it and we were supposed to present together. There's a panel on Ethics in AI in city_l that sounds important, but it conflicts with my teaching schedule. My graduate students are presenting at a student conference in city_m, but I don't need accommodation for that since it's local. I should also mention that my ex-advisor invited me to a reunion dinner in city_n, but that's more social than academic. I'll be traveling alone for all conferences.",
                "goal": "Find accommodation for confirmed academic conferences with specific dates",
                "hard_constraints": {
                    "location_date_pairs": [
                        ("city_a", "2024-06-10 to 2024-06-13"),  # Computational Linguistics
                        ("city_e", "2024-06-12 to 2024-06-14"),  # Cognitive Modeling
                        ("city_f", "2024-06-15 to 2024-06-18"),  # ACL meeting
                        ("city_h", "2024-06-20 to 2024-06-22")   # Neural Information Processing
                    ],
                    "required_locations": ["city_a", "city_e", "city_f", "city_h"],
                    "required_dates": ["2024-06-10 to 2024-06-13", "2024-06-12 to 2024-06-14", "2024-06-15 to 2024-06-18", "2024-06-20 to 2024-06-22"],
                    "guests": 1
                },
                "expected_searches": 4,  # Each conference has specific dates
                "difficulty": "very_hard_with_distractors"
            },
            
            {
                "name": "Multi-City Job Interview Marathon with Scheduling Complexity",
                "client_story": "I'm in the final stages of job hunting after completing my MBA, and I have multiple interviews lined up across different cities. It's both exciting and terrifying! The consulting firm McKinsey wants to fly me out to their city_c office for final rounds, and they've given me flexibility - I can come any time during the week of April 15-19, or the following week April 22-26. They mentioned they could also accommodate me April 29-May 3 if needed. Google has invited me to their city_e headquarters for an on-site interview, and their recruiter said any time from April 20-24 would work, or they could do April 27-May 1. There's also a startup in city_f that I'm really excited about - they're developing AI for healthcare, which aligns perfectly with my thesis work. They want me to visit during their 'culture week' which is April 25-29, but they said I could also come April 18-22 if that works better with my schedule. I had initially planned to interview with Amazon in city_g, but they've put their hiring on hold indefinitely. Goldman Sachs reached out about a position in city_h, but the salary range is too low for my student loans. There's a fintech company in city_i that seemed promising, but after researching their leadership, I don't think it's a good cultural fit. My friend works at a company in city_j and said they might have an opening, but nothing concrete has materialized. I also applied to several companies in city_k, but haven't heard back - their loss! My parents keep suggesting I look at opportunities in city_l where they live, but I want to establish my career independently first. The career services office at my university mentioned some opportunities in city_m, but those are mostly for undergraduates. I'm planning to fly to each city for 2-3 days to allow for multiple interview rounds and to explore the area. I'll be traveling alone for all interviews.",
                "goal": "Find accommodation for confirmed job interviews with company-specific scheduling windows",
                "hard_constraints": {
                    "location_date_pairs": [
                        ("city_c", "2024-04-15 to 2024-04-19"),  # McKinsey option 1
                        ("city_c", "2024-04-22 to 2024-04-26"),  # McKinsey option 2
                        ("city_c", "2024-04-29 to 2024-05-03"),  # McKinsey option 3
                        ("city_e", "2024-04-20 to 2024-04-24"),  # Google option 1
                        ("city_e", "2024-04-27 to 2024-05-01"),  # Google option 2
                        ("city_f", "2024-04-18 to 2024-04-22"),  # Startup option 1
                        ("city_f", "2024-04-25 to 2024-04-29")   # Startup option 2
                    ],
                    "required_locations": ["city_c", "city_e", "city_f"],
                    "required_dates": ["2024-04-15 to 2024-04-19", "2024-04-22 to 2024-04-26", "2024-04-29 to 2024-05-03", "2024-04-20 to 2024-04-24", "2024-04-27 to 2024-05-01", "2024-04-18 to 2024-04-22", "2024-04-25 to 2024-04-29"],
                    "guests": 1
                },
                "expected_searches": 7,  # McKinsey: 3 windows, Google: 2 windows, Startup: 2 windows
                "difficulty": "very_hard_with_distractors"
            },
            
            {
                "name": "Extended Family Wedding Season with Complex Logistics",
                "client_story": "This spring is absolutely chaotic with family events, and I'm trying to plan accommodations for what feels like a never-ending wedding season. My cousin Sarah is getting married in city_a on May 4-6, and as a bridesmaid, I absolutely cannot miss that. My childhood friend Emma is having her wedding in city_d on May 11-13, and I promised to be there since I was her maid of honor. My brother's best friend is getting married in city_e on May 18-20, and since I've known him since we were kids, I really want to attend. There's also my work colleague's wedding in city_f on May 25-27, and it would be good for office politics to show up. My sorority sister is having a destination wedding in city_g on June 1-3, and she's been planning it for two years, so I feel obligated to go. Now, here's where it gets complicated - I might need to adjust some dates because my grandmother's health isn't great and I may need to visit her in between events. For Sarah's wedding, I could arrive as early as May 2 if needed for the rehearsal, staying through May 6. Emma's wedding is more flexible - I could come May 11-13 for just the main event. My brother's friend said I could come May 18-20 for just the wedding. The work colleague's wedding could be May 25-27 for just the wedding weekend. The sorority wedding is pretty firm on June 1-3, but that's non-negotiable. I should mention I was also invited to my ex-roommate's wedding in city_h, but we had a falling out over some drama, so I won't be attending that one. My high school teacher is getting married in city_i, but I barely keep in touch with her, so I'll just send a gift. There's also a wedding in city_j for someone I met at a conference, but I don't know them well enough to justify the travel. My neighbor mentioned her daughter's wedding in city_k, but I think she was just being polite in mentioning it. I'll be traveling with my partner for all the weddings I'm actually attending.",
                "goal": "Find accommodation for confirmed family/friend weddings with specific dates",
                "hard_constraints": {
                    "location_date_pairs": [
                        ("city_a", "2024-05-02 to 2024-05-06"),  # Sarah's wedding
                        ("city_d", "2024-05-11 to 2024-05-13"),  # Emma's wedding
                        ("city_e", "2024-05-18 to 2024-05-20"),  # Brother's friend wedding
                        ("city_f", "2024-05-25 to 2024-05-27"),  # Work colleague wedding
                        ("city_g", "2024-06-01 to 2024-06-03")   # Sorority sister wedding
                    ],
                    "required_locations": ["city_a", "city_d", "city_e", "city_f", "city_g"],
                    "required_dates": ["2024-05-02 to 2024-05-06", "2024-05-11 to 2024-05-13", "2024-05-18 to 2024-05-20", "2024-05-25 to 2024-05-27", "2024-06-01 to 2024-06-03"],
                    "guests": 2
                },
                "expected_searches": 5,  # Each wedding has specific dates
                "difficulty": "extremely_hard_with_distractors"
            },
            
            {
                "name": "Business Development Road Trip with Client Scheduling Constraints",
                "client_story": "I'm a business development manager planning a crucial client visit tour across multiple cities, and I need to coordinate with multiple stakeholders. Our biggest client, TechCorp, wants to meet in city_c during March 20-24. InnovateLabs, our second-largest client, is based in city_e and they want to do a deep-dive strategy session during March 25-29. There's also a potential new client, StartupX, in city_f that I'm really excited about - they could become our biggest account if we land them. They want to meet March 27-31. Finally, there's GrowthCo in city_h that's been asking for a in-person meeting for months. They're only available April 1-5 due to their fiscal year-end scheduling. I should mention that there are several other potential clients I won't be visiting this trip. MegaCorp in city_i keeps rescheduling our meetings, so I've given up on them for now. There's a company in city_j that seemed interested, but they've gone silent on communications. I had a lead in city_k, but they decided to go with a competitor. My boss suggested I visit a company in city_l, but they're not in our target market. There's also a referral opportunity in city_m, but the contact person left the company. I was considering a company in city_n, but their budget is too small for our services. The industry conference in city_o would be great for networking, but it conflicts with these client meetings. My former colleague started a company in city_p, but they're not ready for our services yet. I'm planning to spend 2-3 days in each city to allow for multiple meetings and relationship building. I'll be traveling alone for all client visits.",
                "goal": "Find accommodation for confirmed client meetings with specific scheduling windows",
                "hard_constraints": {
                    "location_date_pairs": [
                        ("city_c", "2024-03-20 to 2024-03-24"),  # TechCorp meeting
                        ("city_e", "2024-03-25 to 2024-03-29"),  # InnovateLabs meeting
                        ("city_f", "2024-03-27 to 2024-03-31"),  # StartupX meeting
                        ("city_h", "2024-04-01 to 2024-04-05")   # GrowthCo meeting
                    ],
                    "required_locations": ["city_c", "city_e", "city_f", "city_h"],
                    "required_dates": ["2024-03-20 to 2024-03-24", "2024-03-25 to 2024-03-29", "2024-03-27 to 2024-03-31", "2024-04-01 to 2024-04-05"],
                    "guests": 1
                },
                "expected_searches": 4,  # Each client has specific meeting dates
                "difficulty": "extremely_hard_with_distractors"
            },
            
            {
                "name": "Medical Specialist Consultation Tour with Appointment Flexibility",
                "client_story": "I'm dealing with a complex medical condition that requires consultations with multiple specialists across different cities, and coordinating these appointments is proving to be incredibly challenging. My primary care doctor has referred me to several specialists, and I need to see them all within the next few months. The rheumatologist at Johns Hopkins in city_c has scheduled me for April 8-12. The endocrinologist at Mayo Clinic in city_e has scheduled me for April 15-19. There's also a clinical trial I'm considering at a research hospital in city_f, and the principal investigator wants to meet with me April 22-26 to discuss eligibility. I should mention there are several other medical options I'm not pursuing. There's a specialist in city_g that my aunt recommended, but they're not covered by my insurance. My neighbor suggested a doctor in city_h, but they don't specialize in my specific condition. I found a clinic in city_i online that looked promising, but when I called, they said they're not accepting new patients. There's a famous doctor in city_j that I've seen on TV, but their approach seems too alternative for my comfort level. My insurance company suggested a provider in city_k, but the reviews are terrible. I was considering a specialist in city_l, but they're retiring next month. There's also a clinical trial in city_m that I applied for, but I didn't meet the criteria. My cousin mentioned a doctor in city_n who helped her, but she has a completely different condition. I found a support group that meets in city_o, but that's more for emotional support than medical treatment. I'm planning to stay 2-3 days in each city to allow for multiple appointments and follow-up tests. I'll be traveling alone for all medical appointments.",
                "goal": "Find accommodation for confirmed medical specialist consultations with specific appointment dates",
                "hard_constraints": {
                    "location_date_pairs": [
                        ("city_c", "2024-04-08 to 2024-04-12"),  # Johns Hopkins rheumatologist
                        ("city_e", "2024-04-15 to 2024-04-19"),  # Mayo Clinic endocrinologist
                        ("city_f", "2024-04-22 to 2024-04-26")   # Research hospital clinical trial
                    ],
                    "required_locations": ["city_c", "city_e", "city_f"],
                    "required_dates": ["2024-04-08 to 2024-04-12", "2024-04-15 to 2024-04-19", "2024-04-22 to 2024-04-26"],
                    "guests": 1
                },
                "expected_searches": 3,  # Each specialist has specific appointment dates
                "difficulty": "extremely_hard_with_distractors"
            },
            
            {
                "name": "Flexible Weekend Getaway with Variable Duration",
                "client_story": "My partner and I want to take a short weekend trip, but we're flexible on the exact dates and duration. We're looking for 2-4 days during the period March 15-22. We could go to either city_a for the museums or city_b for the outdoors. We want to compare all our options to find the best deals and see what works with our schedules. The flexibility is important because we're not sure how much time we can get off work.",
                "goal": "Find accommodation for flexible duration weekend trip with all possible date combinations",
                "hard_constraints": {
                    "required_locations": ["city_a", "city_b"],
                    "required_dates": [
                        # All 2-day combinations during March 15-22
                        "2024-03-15 to 2024-03-16", "2024-03-16 to 2024-03-17", "2024-03-17 to 2024-03-18", 
                        "2024-03-18 to 2024-03-19", "2024-03-19 to 2024-03-20", "2024-03-20 to 2024-03-21", 
                        "2024-03-21 to 2024-03-22",
                        # All 3-day combinations
                        "2024-03-15 to 2024-03-17", "2024-03-16 to 2024-03-18", "2024-03-17 to 2024-03-19", 
                        "2024-03-18 to 2024-03-20", "2024-03-19 to 2024-03-21", "2024-03-20 to 2024-03-22",
                        # All 4-day combinations  
                        "2024-03-15 to 2024-03-18", "2024-03-16 to 2024-03-19", "2024-03-17 to 2024-03-20", 
                        "2024-03-18 to 2024-03-21", "2024-03-19 to 2024-03-22"
                    ],
                    "guests": 2
                },
                "expected_searches": 36,  # 2 locations × 18 date combinations (7+6+5)
                "difficulty": "variable_window"
            },
            
            {
                "name": "Short Business Trip with Flexible Timing",
                "client_story": "I need to visit our regional office for a quick business trip. I'm flexible on whether this is a 2-day or 3-day visit during the window of April 10-15. The meetings could be condensed into 2 days if needed, or spread over 3 days for a more relaxed schedule. I need to check accommodation options in city_c where the office is located.",
                "goal": "Find accommodation for flexible business trip duration with all possible date combinations",
                "hard_constraints": {
                    "required_locations": ["city_c"],
                    "required_dates": [
                        # All 2-day combinations during April 10-15
                        "2024-04-10 to 2024-04-11", "2024-04-11 to 2024-04-12", "2024-04-12 to 2024-04-13", 
                        "2024-04-13 to 2024-04-14", "2024-04-14 to 2024-04-15",
                        # All 3-day combinations
                        "2024-04-10 to 2024-04-12", "2024-04-11 to 2024-04-13", "2024-04-12 to 2024-04-14", 
                        "2024-04-13 to 2024-04-15"
                    ],
                    "guests": 1
                },
                "expected_searches": 9,  # 1 location × 9 date combinations (5+4)
                "difficulty": "variable_window"
            },
            
            {
                "name": "Quick City Break with Duration Choice",
                "client_story": "I want to take a quick city break to city_d and I'm trying to decide between a 2-day or 3-day stay during the long weekend of May 18-20. I could do a short 2-day trip or extend it to 3 days if I can get Monday off work. I want to see what accommodation options are available for both durations during this specific weekend period.",
                "goal": "Find accommodation for city break with choice between 2-day and 3-day durations",
                "hard_constraints": {
                    "required_locations": ["city_d"],
                    "required_dates": [
                        # 2-day option (one combination that fits in the weekend)
                        "2024-05-18 to 2024-05-19",
                        # 3-day option (one combination that covers the long weekend)
                        "2024-05-18 to 2024-05-20"
                    ],
                    "guests": 1
                },
                "expected_searches": 2,  # 1 location × 2 date combinations
                "difficulty": "variable_window"
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            task = Task(
                task_id=f"hotel_booking_{i+1}",
                name=f"Hotel Booking: {scenario['name']}",
                description=scenario['goal'],
                prompt=self._create_clean_prompt(scenario),
                expected_output="SEARCH_PLAN_COMPLETE",
                evaluation_criteria={
                    "search_space_coverage": True,
                    "constraint_inference": True
                },
                metadata={
                    "scenario": scenario['name'],
                    "difficulty": scenario['difficulty'], 
                    "hard_constraints": scenario['hard_constraints'],
                    "expected_searches": scenario['expected_searches'],
                    "goal": scenario['goal'],
                    "client_story": scenario['client_story']
                }
            )
            tasks.append(task)
        
        return tasks
    
    def _create_clean_prompt(self, scenario: Dict[str, Any]) -> str:
        """Create clean prompt with client story and consistent examples."""
        
        return f"""HOTEL BOOKING REQUEST

CLIENT REQUEST:
{scenario['client_story']}

GOAL:
{scenario['goal']}

TASK: 
Create a systematic search plan that covers the minimum required search space to satisfy this client's needs. Analyze the client's story to identify the essential locations and dates that must be searched.

SEARCH INTERFACE:
You can search hotels using only these parameters:
- location (city name)
- checkin_date (YYYY-MM-DD format)  
- checkout_date (YYYY-MM-DD format)
- guests (number of people)

EXAMPLE OUTPUT:
Step 1: location=city_a, dates=2024-03-15 to 2024-03-17, guests=2
Step 2: location=city_b, dates=2024-03-15 to 2024-03-17, guests=2
Step 3: location=city_c, dates=2024-03-20 to 2024-03-22, guests=2

IMPORTANT CONSTRAINTS:
- Output ONLY the search steps in the exact format shown above
- Do NOT include explanations, reasoning, or additional text
- Do NOT include "SEARCH PLAN:" header or any other text
- Each step must follow the format: Step X: location=Y, dates=Z to W, guests=N
- Use concrete values - no variables or placeholders
- When a fixed date range is mentioned (e.g., "March 15-17"), book the ENTIRE range exactly as specified
- When a variable window is requested (e.g., "3-5 days during March 15-21"), search ALL possible date combinations for each window length
- Match the full date windows from the client story exactly

OUTPUT THE SEARCH STEPS ONLY:"""
    
    def _parse_search_plan(self, response_text: str) -> SearchPlan:
        """Parse model response to extract search plan."""
        steps = []
        
        if not response_text:
            return SearchPlan(steps=steps)
        
        # Parse search steps with "Step X:" prefix
        step_pattern = r'Step\s*(\d+):\s*location=([^,]+),\s*dates=([^,]+)\s+to\s+([^,]+),\s*guests=(\d+)'
        matches = re.findall(step_pattern, response_text, re.IGNORECASE)
        
        for match in matches:
            step_num, location, checkin, checkout, guests = match
            try:
                steps.append(SearchStep(
                    step_number=int(step_num),
                    location=location.strip(),
                    checkin=checkin.strip(),
                    checkout=checkout.strip(),
                    guests=int(guests)
                ))
            except ValueError:
                continue  # Skip malformed steps
        
        # If no steps found with "Step X:" prefix, try without prefix - dates format
        if not steps:
            no_step_pattern = r'location=([^,]+),\s*dates=([^,]+)\s+to\s+([^,]+),\s*guests=(\d+)'
            matches = re.findall(no_step_pattern, response_text, re.IGNORECASE)
            
            for i, match in enumerate(matches, 1):
                location, checkin, checkout, guests = match
                try:
                    steps.append(SearchStep(
                        step_number=i,
                        location=location.strip(),
                        checkin=checkin.strip(),
                        checkout=checkout.strip(),
                        guests=int(guests)
                    ))
                except ValueError:
                    continue  # Skip malformed steps
        
        # If still no steps found, try checkin_date/checkout_date format
        if not steps:
            checkin_pattern = r'location=([^,]+),\s*checkin_date=([^,]+),\s*checkout_date=([^,]+),\s*guests=(\d+)'
            matches = re.findall(checkin_pattern, response_text, re.IGNORECASE)
            
            for i, match in enumerate(matches, 1):
                location, checkin, checkout, guests = match
                try:
                    steps.append(SearchStep(
                        step_number=i,
                        location=location.strip(),
                        checkin=checkin.strip(),
                        checkout=checkout.strip(),
                        guests=int(guests)
                    ))
                except ValueError:
                    continue  # Skip malformed steps
        
        # If still no steps found, try "Step X:" with checkin_date/checkout_date format
        if not steps:
            step_checkin_pattern = r'Step\s*(\d+):\s*location=([^,]+),\s*checkin_date=([^,]+),\s*checkout_date=([^,]+),\s*guests=(\d+)'
            matches = re.findall(step_checkin_pattern, response_text, re.IGNORECASE)
            
            for match in matches:
                step_num, location, checkin, checkout, guests = match
                try:
                    steps.append(SearchStep(
                        step_number=int(step_num),
                        location=location.strip(),
                        checkin=checkin.strip(),
                        checkout=checkout.strip(),
                        guests=int(guests)
                    ))
                except ValueError:
                    continue  # Skip malformed steps
        
        return SearchPlan(steps=steps)
    
    def _calculate_coverage_score(self, plan: SearchPlan, hard_constraints: HardConstraints) -> Dict[str, float]:
        """Calculate Coverage × Efficiency score for search space planning."""
        if not plan.steps:
            return {"coverage": 0.0, "efficiency": 0.0, "final_score": 0.0, "deviation_ratio": 0.0, "expected_searches": 0, "actual_searches": 0}
        
        # Get required combinations from hard constraints
        required_combinations = hard_constraints.get_required_combinations()
        required_set = set()
        
        for location, date_range, guests in required_combinations:
            required_set.add((location, date_range, guests))
        
        # Get actual combinations searched
        searched_set = set()
        for step in plan.steps:
            date_range = f"{step.checkin} to {step.checkout}"
            searched_set.add((step.location, date_range, step.guests))
        
        # Calculate coverage: what fraction of required combinations did they search?
        if not required_set:
            coverage = 1.0 if len(searched_set) == 0 else 0.0
        else:
            covered = required_set.intersection(searched_set)
            coverage = len(covered) / len(required_set)
        
        # Calculate efficiency: how close to optimal search count?
        expected_searches = len(required_set)
        actual_searches = len(plan.steps)
        
        # Initialize deviation_ratio
        deviation_ratio = 0.0
        
        if expected_searches > 0:
            deviation_ratio = abs(actual_searches - expected_searches) / expected_searches
            # Same efficiency calculation as previous benchmark (0.3 factor)
            efficiency = max(0.1, 1.0 - (deviation_ratio * 0.3))
        else:
            efficiency = 1.0 if actual_searches == 0 else 0.0
        
        # Final score: Coverage × Efficiency (same as previous benchmark)
        final_score = coverage * efficiency
        
        return {
            "coverage": coverage,
            "efficiency": efficiency, 
            "final_score": min(1.0, final_score),
            "deviation_ratio": deviation_ratio,
            "expected_searches": expected_searches,
            "actual_searches": actual_searches
        }
    
    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a clean hotel booking task - pure search space planning."""
        start_time = time.time()
        
        try:
            # Generate model response
            model_response = await model.generate(task.prompt)
            execution_time = float(model_response.latency or 0.0)
            
            # Parse search plan
            parsed_plan = self._parse_search_plan(model_response.text)
            
            # Calculate coverage score (Coverage × Efficiency) - no execution needed
            hard_constraints = HardConstraints(
                required_locations=task.metadata['hard_constraints']['required_locations'],
                required_dates=task.metadata['hard_constraints']['required_dates'],
                guests=task.metadata['hard_constraints']['guests'],
                location_date_pairs=task.metadata['hard_constraints'].get('location_date_pairs')
            )
            
            score_breakdown = self._calculate_coverage_score(parsed_plan, hard_constraints)
            final_score = score_breakdown["final_score"]
            
            # We only care about score, not arbitrary success thresholds
            success = True
            
            # Detailed metrics
            metrics = {
                "coverage": score_breakdown["coverage"],
                "efficiency": score_breakdown["efficiency"], 
                "final_score": final_score,
                "deviation_ratio": score_breakdown["deviation_ratio"],
                "expected_searches": score_breakdown["expected_searches"],
                "actual_searches": score_breakdown["actual_searches"],
                "searches_planned": len(parsed_plan.steps),
                "output_tokens": model_response.completion_tokens if model_response else 0,
                "constraint_inference_success": score_breakdown["coverage"] > 0.0,
                "planning_quality": "perfect" if final_score >= 0.95 else 
                                  "good" if final_score >= 0.7 else
                                  "fair" if final_score >= 0.3 else "poor"
            }
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=success,
                score=final_score,  # Score is Coverage × Efficiency
                metrics=metrics,
                model_response=model_response,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {str(e)}")
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,  # Only actual errors are failures
                score=0.0,
                metrics={"error": str(e)},
                model_response=None,
                execution_time=0.0
            )
    
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Calculate score for a task response."""
        # This is handled in evaluate_task for consistency
        return 0.0 