import taco_bindings

# Using the more computationally intensive "spmv" example
s = taco_bindings.ScheduleEnv("spmv")
print(s)

print(s.statement())

# Removed the reordering as it's not compatible with the spmv structure
s.split("j", "a", "b", 8)
# s.reorder(["j", "i"])
# s.fuse("i", "j", "f")

print(s.statement())

time = s.execute()
print(f"took {time}ms")

# print(s.code())
