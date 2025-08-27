"""Query parser for understanding temporal and relationship queries."""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from ..models.clustering import QueryIntent


class QueryParser:
    """Parses natural language queries to extract intent and filters."""
    
    def __init__(self):
        # Temporal patterns
        self.temporal_patterns = {
            'first_time': [
                r'first time.*?(?:talked|discussed|mentioned|said).*?about\s+(\w+)',
                r'when.*?first.*?(?:talked|discussed|mentioned|said).*?about\s+(\w+)',
                r'earliest.*?(?:conversation|message|mention).*?about\s+(\w+)'
            ],
            'time_period': [
                r'in\s+(\d{4})',  # in 2018
                r'during\s+(\d{4})',  # during 2018
                r'(\d{4})',  # 2018
                r'last\s+(week|month|year)',  # last week
                r'this\s+(week|month|year)',  # this week
                r'past\s+(week|month|year)',  # past week
                r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago'  # 5 days ago
            ],
            'frequency': [
                r'most\s+(?:talked|messaged|contacted|frequent)',
                r'who.*?talked\s+to\s+most',
                r'most\s+active.*?(?:contact|friend|person)',
                r'frequent.*?(?:contact|friend|person)'
            ],
            'pattern': [
                r'how.*?evolved',
                r'pattern.*?(?:conversation|communication)',
                r'trend.*?(?:conversation|communication)',
                r'change.*?(?:relationship|communication)'
            ]
        }
        
        # Content patterns
        self.content_patterns = {
            'topic': [
                r'about\s+(\w+)',
                r'discussed\s+(\w+)',
                r'talked\s+about\s+(\w+)',
                r'mention.*?(\w+)',
                r'(\w+)\s+conversation'
            ]
        }
        
        # Relationship patterns
        self.relationship_patterns = {
            'participant': [
                r'with\s+(\w+)',
                r'to\s+(\w+)',
                r'(\w+)\s+(?:and|&)\s+(\w+)',  # John and Jane
                r'friend.*?(\w+)',
                r'contact.*?(\w+)'
            ]
        }
        
        # Platform patterns
        self.platform_patterns = {
            'platform': [
                r'on\s+(gmail|email|messenger|whatsapp|instagram|facebook)',
                r'via\s+(gmail|email|messenger|whatsapp|instagram|facebook)',
                r'(gmail|email|messenger|whatsapp|instagram|facebook)'
            ]
        }
    
    def parse_query(self, query: str) -> QueryIntent:
        """Parse a natural language query to extract intent and filters."""
        query_lower = query.lower()
        
        # Initialize query intent
        intent = QueryIntent(
            temporal_filters=[],
            content_filters=[],
            relationship_filters=[],
            platform_filters=[],
            query_type="general",
            confidence=0.0
        )
        
        # Determine query type
        query_type = self._determine_query_type(query_lower)
        intent.query_type = query_type
        
        # Extract temporal filters
        temporal_filters = self._extract_temporal_filters(query_lower)
        intent.temporal_filters = temporal_filters
        
        # Extract content filters
        content_filters = self._extract_content_filters(query_lower)
        intent.content_filters = content_filters
        
        # Extract relationship filters
        relationship_filters = self._extract_relationship_filters(query_lower)
        intent.relationship_filters = relationship_filters
        
        # Extract platform filters
        platform_filters = self._extract_platform_filters(query_lower)
        intent.platform_filters = platform_filters
        
        # Calculate confidence based on how well we parsed the query
        confidence = self._calculate_confidence(intent, query_lower)
        intent.confidence = confidence
        
        logger.debug(f"Parsed query: {query} -> {intent.dict()}")
        
        return intent
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query being asked."""
        if any(re.search(pattern, query) for pattern in self.temporal_patterns['first_time']):
            return "first_occurrence"
        elif any(re.search(pattern, query) for pattern in self.temporal_patterns['frequency']):
            return "most_frequent"
        elif any(re.search(pattern, query) for pattern in self.temporal_patterns['pattern']):
            return "pattern_analysis"
        elif any(re.search(pattern, query) for pattern in self.temporal_patterns['time_period']):
            return "temporal_analysis"
        else:
            return "general_search"
    
    def _extract_temporal_filters(self, query: str) -> List[Dict[str, Any]]:
        """Extract temporal filters from query."""
        filters = []
        
        # Check for specific years
        year_matches = re.findall(r'(\d{4})', query)
        for year in year_matches:
            filters.append({
                'type': 'year',
                'value': int(year),
                'start_date': datetime(int(year), 1, 1),
                'end_date': datetime(int(year), 12, 31, 23, 59, 59)
            })
        
        # Check for relative time periods
        if 'last week' in query:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            filters.append({
                'type': 'relative',
                'value': 'last_week',
                'start_date': start_date,
                'end_date': end_date
            })
        elif 'last month' in query:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            filters.append({
                'type': 'relative',
                'value': 'last_month',
                'start_date': start_date,
                'end_date': end_date
            })
        elif 'last year' in query:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            filters.append({
                'type': 'relative',
                'value': 'last_year',
                'start_date': start_date,
                'end_date': end_date
            })
        
        # Check for "ago" patterns
        ago_matches = re.findall(r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago', query)
        for amount, unit in ago_matches:
            amount = int(amount)
            if 'day' in unit:
                delta = timedelta(days=amount)
            elif 'week' in unit:
                delta = timedelta(weeks=amount)
            elif 'month' in unit:
                delta = timedelta(days=amount * 30)
            elif 'year' in unit:
                delta = timedelta(days=amount * 365)
            else:
                continue
            
            end_date = datetime.now() - delta
            start_date = end_date - delta  # Look at the period before that
            filters.append({
                'type': 'relative',
                'value': f'{amount}_{unit}_ago',
                'start_date': start_date,
                'end_date': end_date
            })
        
        return filters
    
    def _extract_content_filters(self, query: str) -> List[str]:
        """Extract content/topic filters from query."""
        filters = []
        
        # Look for topic mentions
        for pattern in self.content_patterns['topic']:
            matches = re.findall(pattern, query)
            filters.extend(matches)
        
        # Look for specific topics in first_time patterns
        for pattern in self.temporal_patterns['first_time']:
            matches = re.findall(pattern, query)
            filters.extend(matches)
        
        return list(set(filters))  # Remove duplicates
    
    def _extract_relationship_filters(self, query: str) -> List[str]:
        """Extract relationship/participant filters from query."""
        filters = []
        
        # Look for participant mentions
        for pattern in self.relationship_patterns['participant']:
            matches = re.findall(pattern, query)
            # Handle tuples (e.g., "John and Jane")
            for match in matches:
                if isinstance(match, tuple):
                    filters.extend(match)
                else:
                    filters.append(match)
        
        return list(set(filters))  # Remove duplicates
    
    def _extract_platform_filters(self, query: str) -> List[str]:
        """Extract platform filters from query."""
        filters = []
        
        for pattern in self.platform_patterns['platform']:
            matches = re.findall(pattern, query)
            filters.extend(matches)
        
        return list(set(filters))  # Remove duplicates
    
    def _calculate_confidence(self, intent: QueryIntent, query: str) -> float:
        """Calculate confidence score for the parsed query."""
        confidence = 0.0
        
        # Base confidence
        if intent.query_type != "general_search":
            confidence += 0.3
        
        # Temporal filters
        if intent.temporal_filters:
            confidence += 0.2
        
        # Content filters
        if intent.content_filters:
            confidence += 0.2
        
        # Relationship filters
        if intent.relationship_filters:
            confidence += 0.2
        
        # Platform filters
        if intent.platform_filters:
            confidence += 0.1
        
        # Query complexity bonus
        if len(intent.temporal_filters) + len(intent.content_filters) + len(intent.relationship_filters) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_query_suggestions(self) -> List[str]:
        """Get example queries to help users."""
        return [
            "When was the first time I talked about Berkeley?",
            "Who did I talk to most in 2018?",
            "What were my most active conversation topics last month?",
            "How did my communication with John evolve over time?",
            "What did I discuss most frequently on WhatsApp?",
            "Who was my most frequent contact in 2020?",
            "When did I start talking about machine learning?",
            "What were the main topics of my conversations last year?"
        ]
