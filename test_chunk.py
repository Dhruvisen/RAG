import nltk
table = """| Sr. No. | Part No. / Rev. No. | Description | HSN Code | Delivery Date | Order Qty PUOM | Rate (INR) | Amount (INR) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | STNRY 000002 Rev: 0 | TOURCH SMALL | 85131010 | 03-07-2026 | 2.0000 NO | 230.00 | 460.00 |
| 2 | STNRY 000005 Rev: 0 | PEN BLUE (FLAIR SPRING BALL PEN) | 96081019 | 10-07-2026 | 40.0000 NO | 3.40 | 136.00 |"""

sentences = nltk.sent_tokenize(table)
print(f"Total sentences: {len(sentences)}")
for i, s in enumerate(sentences):
    print(f"S{i}: {s}")
