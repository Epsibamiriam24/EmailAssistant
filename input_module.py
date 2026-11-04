from __future__ import annotations


def collect_customer_email() -> dict[str, str]:
    print("\nEnter customer details")
    name = input("Customer name: ").strip()
    subject = input("Email subject: ").strip()
    print("Customer email (enter a blank line when finished):")
    lines: list[str] = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)

    email = " ".join(lines).strip()
    if not email:
        raise ValueError("Customer email cannot be blank.")
    return {"name": name or "Customer", "subject": subject or "Customer enquiry", "email": email}
