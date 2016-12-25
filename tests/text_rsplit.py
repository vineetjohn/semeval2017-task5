sample_string = \
    "According to Gran , the company has no plans to move all production to Russia" \
    " , although that is where the company is growing .@neutral"

print sample_string.rsplit(None, 1)[0]
print sample_string.rsplit(None, 1)[-1]

