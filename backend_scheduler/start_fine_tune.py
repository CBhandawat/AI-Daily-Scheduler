from openai import OpenAI

client = OpenAI()

# List all fine-tuning jobs
jobs = client.fine_tuning.jobs.list()

for job in jobs:
    print("====")
    print("Job ID:", job.id)
    print("Status:", job.status)
    print("Model:", job.fine_tuned_model)
