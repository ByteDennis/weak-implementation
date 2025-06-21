def start_cli():
    from weakly.cli.main import main
    main()
    
class A:
    def __init__(self):
        self.a = self.b = None

def use_upath():
    from upath import UPath
    print((x := UPath("datasets", "test.log")).exists())
    a = A()
    a.b = 1
    print(a.a)

    


if __name__ == "__main__":
    use_upath()
    ...
