"""Data loader for processing JSONL files into normalized conversation models."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import logging

from ..models.conversation import Conversation, Message, Participant
from ..utils.language_detection import detect_language, is_georgian, is_romanized_georgian, normalize_georgian_roman
from ..utils.date_utils import parse_date

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and normalizes conversation data from JSONL files."""
    
    def __init__(self, raw_data_dir: str = "./data/raw", processed_dir: str = "./data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Known user accounts for identifying self
        self.known_accounts = {
            "sandronatchkebia@gmail.com",
            "natchkebia@berkeley.edu",
            "aleksandre_natchkebia"
        }
        
        # Platform-specific handlers
        self.platform_handlers = {
            "gmail": self._process_gmail_message,
            "berkeley_mail": self._process_gmail_message,  # Same format as Gmail
            "messenger": self._process_messenger_message,
            "whatsapp": self._process_whatsapp_message,
            "instagram": self._process_instagram_message,
        }
    
    async def load_all_platforms(self) -> Dict[str, List[Conversation]]:
        """Load and process all platform data."""
        results = {}
        
        for jsonl_file in self.raw_data_dir.glob("*.jsonl"):
            platform = self._extract_platform_from_filename(jsonl_file.name)
            if platform and platform in self.platform_handlers:
                logger.info(f"Processing {platform} data from {jsonl_file.name}")
                conversations = await self._load_platform_data(jsonl_file, platform)
                results[platform] = conversations
                
                # Save processed data
                await self._save_platform_data(platform, conversations)
        
        return results
    
    async def _load_platform_data(self, file_path: Path, platform: str) -> List[Conversation]:
        """Load and process data for a specific platform."""
        conversations = {}
        
        async for message_data in self._stream_jsonl(file_path):
            try:
                message = await self.platform_handlers[platform](message_data)
                if message:
                    conv_id = message_data.get("conversation_id", "unknown")
                    if conv_id not in conversations:
                        conversations[conv_id] = self._create_conversation_skeleton(message_data, platform)
                    
                    conversations[conv_id].messages.append(message)
                    
                    # Update conversation metadata
                    self._update_conversation_metadata(conversations[conv_id], message)
                    
            except Exception as e:
                logger.error(f"Error processing message in {platform}: {e}")
                continue
        
        return list(conversations.values())
    
    async def _stream_jsonl(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Stream JSONL file line by line."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue
    
    def _extract_platform_from_filename(self, filename: str) -> Optional[str]:
        """Extract platform name from filename."""
        if "gmail" in filename:
            return "gmail"
        elif "berkeley" in filename:
            return "berkeley_mail"
        elif "messenger" in filename:
            return "messenger"
        elif "whatsapp" in filename:
            return "whatsapp"
        elif "instagram" in filename:
            return "instagram"
        return None
    
    def _create_conversation_skeleton(self, message_data: Dict[str, Any], platform: str) -> Conversation:
        """Create a conversation skeleton from message data."""
        participants = self._extract_participants(message_data, platform)
        
        return Conversation(
            id=f"{platform}:{message_data.get('conversation_id', 'unknown')}",
            platform=platform,
            participants=participants,
            messages=[],
            start_date=parse_date(message_data.get("date", "1970-01-01T00:00:00Z")),
            last_activity=parse_date(message_data.get("date", "1970-01-01T00:00:00Z")),
            metadata={
                "source_file": message_data.get("source_file", ""),
                "platform_specific": self._extract_platform_metadata(message_data, platform)
            }
        )
    
    def _extract_participants(self, message_data: Dict[str, Any], platform: str) -> List[Participant]:
        """Extract participants from message data."""
        participants = []
        
        if platform in ["gmail", "berkeley_mail"]:
            # Email platforms
            from_email = message_data.get("from", "")
            to_emails = message_data.get("to", [])
            cc_emails = message_data.get("cc", [])
            
            # Add sender
            if from_email:
                participants.append(Participant(
                    id=from_email,
                    name=from_email.split("@")[0],
                    email=from_email,
                    platform_id=from_email,
                    is_self=from_email in self.known_accounts
                ))
            
            # Add recipients
            all_recipients = set(to_emails + cc_emails)
            for email in all_recipients:
                if email and email not in [p.email for p in participants]:
                    participants.append(Participant(
                        id=email,
                        name=email.split("@")[0],
                        email=email,
                        platform_id=email,
                        is_self=email in self.known_accounts
                    ))
        
        elif platform in ["messenger", "whatsapp", "instagram"]:
            # Social platforms
            platform_participants = message_data.get("participants", [])
            for participant_id in platform_participants:
                if participant_id not in [p.id for p in participants]:
                    participants.append(Participant(
                        id=participant_id,
                        name=participant_id,
                        platform_id=participant_id,
                        is_self=participant_id in self.known_accounts
                    ))
        
        return participants
    
    def _extract_platform_metadata(self, message_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Extract platform-specific metadata."""
        metadata = {}
        
        if platform in ["gmail", "berkeley_mail"]:
            metadata.update({
                "subject": message_data.get("subject"),
                "labels": message_data.get("labels", []),
                "account": message_data.get("account"),
                "sender_domain": message_data.get("sender_domain"),
                "is_reply": message_data.get("is_reply", False),
                "thread_index": message_data.get("thread_index", 0)
            })
        
        elif platform in ["messenger", "whatsapp", "instagram"]:
            metadata.update({
                "attachments": message_data.get("attachments", {}),
                "reactions": message_data.get("reactions", []),
                "thread_len": message_data.get("thread_len", 1),
                "turn_id": message_data.get("turn_id"),
                "partner_ids": message_data.get("partner_ids", [])
            })
        
        return metadata
    
    def _update_conversation_metadata(self, conversation: Conversation, message: Message):
        """Update conversation metadata based on new message."""
        if message.timestamp > conversation.last_activity:
            conversation.last_activity = message.timestamp
        
        # Update participants if new ones found
        for participant in message.metadata.get("participants", []):
            if participant not in [p.id for p in conversation.participants]:
                # This would need to be implemented based on your participant model
                pass
    
    async def _process_gmail_message(self, message_data: Dict[str, Any]) -> Optional[Message]:
        """Process Gmail/Berkeley mail message."""
        try:
            # Normalize text content
            content = message_data.get("body_text") or message_data.get("body_raw", "")
            content = self._normalize_text_content(content)
            
            # Detect language
            lang = message_data.get("lang", "unknown")
            if lang == "unknown":
                lang = detect_language(content)
            
            # Determine sender
            from_email = message_data.get("from", "")
            sender_id = from_email
            
            return Message(
                id=f"{message_data.get('platform', 'gmail')}:{message_data.get('message_id', 'unknown')}",
                content=content,
                sender_id=sender_id,
                timestamp=parse_date(message_data.get("date", "1970-01-01T00:00:00Z")),
                platform=message_data.get("platform", "gmail"),
                message_type=message_data.get("content_type", "text/plain"),
                language=lang,
                metadata={
                    "subject": message_data.get("subject"),
                    "direction": message_data.get("direction"),
                    "turn_role": message_data.get("turn_role"),
                    "is_reply": message_data.get("is_reply", False),
                    "thread_index": message_data.get("thread_index", 0)
                }
            )
        except Exception as e:
            logger.error(f"Error processing Gmail message: {e}")
            return None
    
    async def _process_messenger_message(self, message_data: Dict[str, Any]) -> Optional[Message]:
        """Process Messenger message."""
        try:
            content = message_data.get("body_text") or message_data.get("body_raw", "")
            content = self._normalize_text_content(content)
            
            lang = message_data.get("lang", "unknown")
            if lang == "unknown":
                lang = detect_language(content)
            
            return Message(
                id=f"messenger:{message_data.get('message_id', 'unknown')}",
                content=content,
                sender_id=message_data.get("from", "unknown"),
                timestamp=parse_date(message_data.get("date", "1970-01-01T00:00:00Z")),
                platform="messenger",
                message_type=message_data.get("content_type", "text"),
                language=lang,
                metadata={
                    "direction": message_data.get("direction"),
                    "turn_role": message_data.get("turn_role"),
                    "is_reply": message_data.get("is_reply", False),
                    "thread_index": message_data.get("thread_index", 0),
                    "thread_len": message_data.get("thread_len", 1)
                }
            )
        except Exception as e:
            logger.error(f"Error processing Messenger message: {e}")
            return None
    
    async def _process_whatsapp_message(self, message_data: Dict[str, Any]) -> Optional[Message]:
        """Process WhatsApp message."""
        try:
            content = message_data.get("body_text") or message_data.get("body_raw", "")
            content = self._normalize_text_content(content)
            
            lang = message_data.get("lang", "unknown")
            if lang == "unknown":
                lang = detect_language(content)
            
            return Message(
                id=f"whatsapp:{message_data.get('message_id', 'unknown')}",
                content=content,
                sender_id=message_data.get("from", "unknown"),
                timestamp=parse_date(message_data.get("date", "1970-01-01T00:00:00Z")),
                platform="whatsapp",
                message_type=message_data.get("content_type", "text"),
                language=lang,
                metadata={
                    "direction": message_data.get("direction"),
                    "turn_role": message_data.get("turn_role"),
                    "is_reply": message_data.get("is_reply", False),
                    "thread_index": message_data.get("thread_index", 0)
                }
            )
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {e}")
            return None
    
    async def _process_instagram_message(self, message_data: Dict[str, Any]) -> Optional[Message]:
        """Process Instagram message."""
        try:
            content = message_data.get("body_text") or message_data.get("body_raw", "")
            content = self._normalize_text_content(content)
            
            lang = message_data.get("lang", "unknown")
            if lang == "unknown":
                lang = detect_language(content)
            
            return Message(
                id=f"instagram:{message_data.get('message_id', 'unknown')}",
                content=content,
                sender_id=message_data.get("from", "unknown"),
                timestamp=parse_date(message_data.get("date", "1970-01-01T00:00:00Z")),
                platform="instagram",
                message_type=message_data.get("content_type", "text"),
                language=lang,
                metadata={
                    "direction": message_data.get("direction"),
                    "turn_role": message_data.get("turn_role"),
                    "is_reply": message_data.get("is_reply", False),
                    "thread_index": message_data.get("thread_index", 0),
                    "thread_len": message_data.get("thread_len", 1)
                }
            )
        except Exception as e:
            logger.error(f"Error processing Instagram message: {e}")
            return None
    
    def _normalize_text_content(self, content: str) -> str:
        """Normalize text content for processing."""
        if not content:
            return ""
        
        # Basic normalization
        content = content.strip()
        
        # Handle Georgian text (both scripts)
        if is_georgian(content):
            # Already in Georgian script
            pass
        elif is_romanized_georgian(content):
            # Check if it's romanized Georgian
            normalized = normalize_georgian_roman(content)
            if normalized != content:
                content = normalized
        
        return content
    
    async def _save_platform_data(self, platform: str, conversations: List[Conversation]):
        """Save processed platform data."""
        platform_dir = self.processed_dir / platform
        platform_dir.mkdir(exist_ok=True)
        
        for conversation in conversations:
            # Save individual conversation
            conv_file = platform_dir / f"{conversation.id}.json"
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(conversation.dict(), f, indent=2, ensure_ascii=False, default=str)
        
        # Save platform summary
        summary = {
            "platform": platform,
            "total_conversations": len(conversations),
            "total_messages": sum(len(conv.messages) for conv in conversations),
            "date_range": {
                "start": min(conv.start_date for conv in conversations).isoformat(),
                "end": max(conv.last_activity for conv in conversations).isoformat()
            },
            "conversation_ids": [conv.id for conv in conversations]
        }
        
        summary_file = platform_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

