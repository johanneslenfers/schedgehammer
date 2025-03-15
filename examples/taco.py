import taco_bindings

s = taco_bindings.ScheduleEnv("mini")

print(s.statement())

# s.split("i", "a", "b", 2)
s.reorder(["j", "i"])
# s.fuse("i", "j", "f")

print(s.statement())

time = s.execute()
print(f"took {time}ms")

print(s.code())
