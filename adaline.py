from random import uniform as randomfloat
import matplotlib.pyplot as plt

# Base de dados fornecida
base_dados = {'s1': [2.215, 0.224, 0.294, 2.327, 2.497, 0.169, 1.274, 1.526, 2.009, 1.759, 1.367, 2.173, 0.856, 2.21, 1.587, 0.35, 1.441, 0.185, 2.764, 1.947],
              's2': [2.063, 1.586, 0.651, 2.932, 2.322, 1.943, 2.428, 0.596, 2.161, 0.342, 0.938, 2.719, 1.904, 1.868, 1.642, 0.84, 0.09, 1.327, 1.149, 1.598],
              't': [-1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1]}

# Função Adaline para treinamento
def adaline(entradas, saidas):
    w = [randomfloat(-0.5, 0.5) for _ in range(len(entradas))]
    b = 0  # bias inicial
    l_rate = 0.01  # taxa de aprendizagem
    tolerancia = 0.001  # condição de parada
    num_ciclos = 100  # número máximo de ciclos
    erros = []  # lista para armazenar o erro quadrático

    for ciclo in range(num_ciclos):
        erro_quadratico_total = 0
        maior_alteracao = 0

        for j in range(len(entradas[0])):
            y_liquido = b
            for i in range(len(entradas)):
                y_liquido += entradas[i][j] * w[i]
            
            erro = saidas[j] - y_liquido
            erro_quadratico_total += erro ** 2

            for i in range(len(entradas)):
                alteracao_peso = l_rate * erro * entradas[i][j]
                w[i] += alteracao_peso
                if abs(alteracao_peso) > maior_alteracao:
                    maior_alteracao = abs(alteracao_peso)
            b += l_rate * erro

        erros.append(erro_quadratico_total)

        if maior_alteracao < tolerancia:
            print(f"Treinamento encerrado no ciclo {ciclo + 1}")
            break

    return w, b, erros

# Função Hebb para testar o modelo
def hebb_rule(entrada, pesos, bias):
    y_liquido = bias
    for i in range(len(pesos)):
        y_liquido += pesos[i] * entrada[i]
    return 1 if y_liquido >= 0 else -1

# Função para calcular a acurácia
def acuracia(y_pred, y_real):
    acertos = sum([1 if y_pred[i] == y_real[i] else 0 for i in range(len(y_real))])
    return acertos / len(y_real) * 100

# Função principal que controla o fluxo de treinamento e teste
def main():
    # Defina se quer treinar ou testar
    treinar = True  # Coloque como False se quiser apenas testar

    if treinar:
        # Treinamento da Adaline
        entradas = [base_dados["s1"], base_dados["s2"]]
        saidas = base_dados["t"]
        w, b, erros = adaline(entradas, saidas)

        # Plotar o erro quadrático total durante o treinamento
        plt.plot(range(1, len(erros) + 1), erros)
        plt.title("Erro Quadrático durante o Treinamento")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Total")
        plt.savefig("erro_quadratico.png")
        plt.show()

        # Salvar os pesos e bias em um arquivo
        with open("pesos_bias.txt", "w") as f:
            f.write(f"Pesos: {w}\n")
            f.write(f"Bias: {b}\n")

        print("Treinamento concluído. Pesos e bias salvos em 'pesos_bias.txt'.")

    else:
        # Testar o modelo treinado
        # Carregar os pesos e bias do arquivo
        with open("pesos_bias.txt", "r") as f:
            linhas = f.readlines()
            pesos = [float(x) for x in linhas[0].strip().split(":")[1].strip()[1:-1].split(",")]
            bias = float(linhas[1].strip().split(":")[1].strip())

        entradas = [base_dados["s1"], base_dados["s2"]]
        saidas = base_dados["t"]

        y_pred = []
        for i in range(len(entradas[0])):
            entrada = [entradas[0][i], entradas[1][i]]
            y = hebb_rule(entrada, pesos, bias)
            y_pred.append(y)
            print(f"Entrada: {entrada}, Saída Esperada: {saidas[i]}, Saída Prevista: {y}")

        # Calcular e exibir a acurácia
        acur = acuracia(y_pred, saidas)
        print(f"Acurácia: {acur}%")

if __name__ == "__main__":
    main()
