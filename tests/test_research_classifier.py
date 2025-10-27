import nodes.research_classifier as rc


class DummyMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeChain:
    def __init__(self, to_return=True):
        self.calls = []
        self.to_return = to_return

    def invoke(self, kwargs):
        # record the kwargs passed and return a configured value
        self.calls.append(kwargs)
        return self.to_return


def test_research_classifier_returns_bool(monkeypatch):
    fake = FakeChain(to_return=True)
    monkeypatch.setattr(rc, "chain", fake)

    state = {"chat_history": [DummyMsg("user", "What's the capital of France?")]}

    result = rc.research_classifier_node(state)

    assert result == {"needs_research": True}


def test_research_classifier_passes_conversation_text(monkeypatch):
    fake = FakeChain(to_return=False)
    monkeypatch.setattr(rc, "chain", fake)

    state = {
        "chat_history": [
            DummyMsg("user", "First"),
            DummyMsg("assistant", "Reply"),
            DummyMsg("user", "Last?"),
        ]
    }

    result = rc.research_classifier_node(state)

    # ensure chain.invoke was called and captured the chat_history
    assert fake.calls, "chain.invoke was not called"
    passed = fake.calls[0]["chat_history"]
    expected = "user: First\nassistant: Reply\nuser: Last?"
    assert passed == expected
    assert result == {"needs_research": False}

