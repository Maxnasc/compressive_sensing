class Bolo_de_aniversario:

    def __init__(self):
        self.sabor_bolo = "Chocolate"
        self.tema = "Programação"
        self.aniversariante = "Max"
        self.idade = 26

    def parabens(self):
        print(f"Feliz aniversário {self.aniversariante}!")


if __name__ == "__main__":
    bolo = Bolo_de_aniversario()
    bolo.parabens()
