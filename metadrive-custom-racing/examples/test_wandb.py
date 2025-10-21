"""Test script to verify WandB is working correctly."""
import wandb
import time

print("🧪 Testing WandB connection...")

# Initialize WandB
wandb.init(
    project="test-wandb-connection",
    name="test-run",
    config={"test": True}
)

print("✅ WandB initialized!")

# Log some test data
for i in range(10):
    wandb.log({
        "test_metric": i * 2,
        "random_value": i ** 0.5,
    }, step=i)
    time.sleep(0.1)

print("📊 Test data logged!")
print(f"🔗 Check your run at: {wandb.run.url}")

wandb.finish()
print("✅ Test complete!")