"""Basic tests for AI Memory project."""

import pytest
from ai_memory.models.conversation import Conversation, Message, Participant
from ai_memory.utils.language_detection import detect_language, is_georgian
from ai_memory.utils.file_utils import ensure_directory


def test_participant_model():
    """Test Participant model creation."""
    participant = Participant(
        id="user123",
        name="John Doe",
        email="john@example.com",
        is_self=False
    )
    assert participant.name == "John Doe"
    assert participant.id == "user123"


def test_message_model():
    """Test Message model creation."""
    message = Message(
        id="msg123",
        content="Hello, world!",
        sender_id="user123",
        timestamp="2023-01-01T12:00:00",
        platform="whatsapp"
    )
    assert message.content == "Hello, world!"
    assert message.platform == "whatsapp"


def test_conversation_model():
    """Test Conversation model creation."""
    participant = Participant(id="user123", name="John Doe")
    message = Message(
        id="msg123",
        content="Hello!",
        sender_id="user123",
        timestamp="2023-01-01T12:00:00",
        platform="whatsapp"
    )
    
    conversation = Conversation(
        id="conv123",
        platform="whatsapp",
        participants=[participant],
        messages=[message],
        start_date="2023-01-01T12:00:00",
        last_activity="2023-01-01T12:00:00"
    )
    
    assert len(conversation.participants) == 1
    assert len(conversation.messages) == 1


def test_language_detection():
    """Test basic language detection."""
    assert detect_language("Hello world") == "en"
    assert is_georgian("გამარჯობა") == True
    assert is_georgian("Hello") == False


def test_file_utils():
    """Test file utility functions."""
    # This should not raise an error
    ensure_directory("./test_dir")
    
    # Clean up
    import shutil
    shutil.rmtree("./test_dir", ignore_errors=True)
