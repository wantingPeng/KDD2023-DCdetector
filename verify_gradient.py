"""
验证：即使loss=0，梯度也可以非零
这个脚本演证DCdetector的对抗性训练机制
"""
import torch

def my_kl_loss(p, q):
    """DCdetector使用的KL散度"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

# 创建简单的测试数据
torch.manual_seed(42)
batch_size = 4
win_size = 10
n_features = 5

# 模拟模型输出：series和prior（使用softmax确保是有效概率分布）
series_raw = torch.randn(batch_size, n_features, win_size, requires_grad=True)
prior_raw = torch.randn(batch_size, n_features, win_size, requires_grad=True)

# 使用softmax确保是有效的概率分布
series = torch.softmax(series_raw, dim=-1)
prior_normalized = torch.softmax(prior_raw, dim=-1)

print("=" * 60)
print("验证DCdetector的对抗性训练机制")
print("=" * 60)

# 计算series_loss（series有梯度，prior被detach）
series_loss = (
    torch.mean(my_kl_loss(series, prior_normalized.detach())) + 
    torch.mean(my_kl_loss(prior_normalized.detach(), series))
)

# 计算prior_loss（prior有梯度，series被detach）  
prior_loss = (
    torch.mean(my_kl_loss(prior_normalized, series.detach())) + 
    torch.mean(my_kl_loss(series.detach(), prior_normalized))
)

print(f"\n1. 损失值：")
print(f"   series_loss = {series_loss.item():.10f}")
print(f"   prior_loss  = {prior_loss.item():.10f}")

# 计算最终loss
loss = prior_loss - series_loss

print(f"\n2. 最终loss = prior_loss - series_loss")
print(f"   loss = {loss.item():.15f}")
print(f"   |loss| < 1e-6? {abs(loss.item()) < 1e-6}")

# 关键：即使loss约等于0，我们仍然可以计算梯度！
print(f"\n3. 计算梯度前的参数梯度状态：")
print(f"   series_raw.grad: {series_raw.grad}")
print(f"   prior_raw.grad: {prior_raw.grad}")

# 反向传播
loss.backward()

print(f"\n4. 计算梯度后（loss.backward()）：")
has_series_grad = series_raw.grad is not None and torch.abs(series_raw.grad).sum() > 1e-6
has_prior_grad = prior_raw.grad is not None and torch.abs(prior_raw.grad).sum() > 1e-6

print(f"   series_raw的梯度是否非零? {has_series_grad}")
print(f"   prior_raw的梯度是否非零? {has_prior_grad}")

if series_raw.grad is not None:
    print(f"\n   series_raw.grad 的统计信息：")
    print(f"      - 绝对值和: {torch.abs(series_raw.grad).sum().item():.8f}")
    print(f"      - 均值: {series_raw.grad.mean().item():.8f}")
    print(f"      - 最大值: {series_raw.grad.max().item():.8f}")
    print(f"      - 最小值: {series_raw.grad.min().item():.8f}")

if prior_raw.grad is not None:
    print(f"\n   prior_raw.grad 的统计信息：")
    print(f"      - 绝对值和: {torch.abs(prior_raw.grad).sum().item():.8f}")
    print(f"      - 均值: {prior_raw.grad.mean().item():.8f}")
    print(f"      - 最大值: {prior_raw.grad.max().item():.8f}")
    print(f"      - 最小值: {prior_raw.grad.min().item():.8f}")

print("\n" + "=" * 60)
print("【关键结论】")
print("=" * 60)
print(f"✓ loss值 ≈ 0（数值上 = {loss.item():.10f}）")
print(f"✓ 但series梯度 {'非零' if has_series_grad else '为零'}！")
print(f"✓ 且prior梯度 {'非零' if has_prior_grad else '为零'}！")
print("\n这是因为：")
print("  1. series_loss 只对 series 有梯度（prior被detach）")
print("  2. prior_loss 只对 prior 有梯度（series被detach）")
print("  3. loss = prior_loss - series_loss")
print("  4. ∂loss/∂series = -∂series_loss/∂series ≠ 0")
print("  5. ∂loss/∂prior = ∂prior_loss/∂prior ≠ 0")
print("\n即使两个损失数值相等（loss=0），")
print("detach()的位置不同导致梯度流向不同！")
print("这就是DCdetector的对抗性训练机制！")
print("=" * 60)

# 模拟参数更新
if has_series_grad and has_prior_grad:
    print("\n5. 模拟参数更新（梯度下降）：")
    learning_rate = 0.01
    with torch.no_grad():
        series_before = series_raw.clone()
        prior_before = prior_raw.clone()
        
        series_raw.sub_(learning_rate * series_raw.grad)
        prior_raw.sub_(learning_rate * prior_raw.grad)
        
        series_change = torch.abs(series_raw - series_before).sum()
        prior_change = torch.abs(prior_raw - prior_before).sum()
        
        print(f"   series_raw参数变化量: {series_change.item():.8f}")
        print(f"   prior_raw参数变化量: {prior_change.item():.8f}")
        print(f"   ✓ 参数确实被更新了！即使loss=0！")

print("\n" + "=" * 60)
