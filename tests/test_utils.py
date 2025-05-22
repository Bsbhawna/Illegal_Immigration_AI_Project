# tests/test_utils.py

from darkweb_monitoring.utils import mask_credit_card

def test_mask_credit_card():
    assert mask_credit_card("1234567812345678") == "**** **** **** 5678"
    assert mask_credit_card("4444-5555-6666-7777") == "**** **** **** 7777"
    assert mask_credit_card("0000 1111 2222 3333") == "**** **** **** 3333"
