import autodiff as ad

x1 =ad.Variable(name="x1")
x2 = ad.Varible(name="x2")

y = x1 * x2 + x1

executor = ad.Executor([y])
y_val = executor.run(feed_dict={x1: x1_val, x2: x2_val})

