class Person:
    def __call__(self, name):
        print("call"+"hello"+ name)

    def hello(self, name):
        print("hello"+ name)

person = Person()
# __call__函数可以直接：对象（参数）来调用
person("hello")
# 其他函数需要用：对象.函数（参数）来调用
person.hello("hello")