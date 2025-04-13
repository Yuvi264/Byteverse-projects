from faker import Faker
import json
import random

fake = Faker()

problems = ["anxiety", "depression", "PTSD", "OCD", "bipolar", "eating disorders", 
            "addiction", "parenting stress", "LGBTQ+ issues", "workplace burnout"]

mentors = []
for i in range(150):
    mentors.append({
        "id": f"mentor{i:03d}",
        "name": fake.name(),
        "problem_tags": random.sample(problems, k=random.randint(1, 3)),
        "story": fake.paragraph(nb_sentences=3),
        "language": random.choice(["hindi", "english", "tamil", "telugu", "bengali"]),
        "rating": round(random.uniform(3.5, 5.0), 1),
        "contact": fake.email(),
        "availability": random.sample(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], k=random.randint(2, 4))
    })

with open('mentors.json', 'w') as f:
    json.dump(mentors, f, indent=2)