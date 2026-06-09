import jax
import jax.numpy as jnp

print(jax.devices())
print(jax.local_devices())

cpu_device = jax.devices('cpu')[0]
print(cpu_device)


x_tpu = jnp.arange(5)
x_cpu = x_tpu.to_device(cpu_device)
# print(isinstance(x, jax.Array))
print(x_tpu.devices())
# print(x.sharding)

print(x_cpu.devices())