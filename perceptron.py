def main():   
    X = [
        1, -1, -1, -1, 1,
        -1, 1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, 1, -1,
        1, -1, -1, -1, 1
    ]

    T = [
        1, 1, 1, 1, 1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1
    ]

    resultado = perceptron([X,T], [1,-1])
    print(neuronio(resultado[0], resultado[1],X)) #Se for X, o resultado que deve sair é 1
    print(neuronio(resultado[0], resultado[1],T)) #Se for T, o resultado que deve sair é -1


def perceptron(letras,saidas): #função perceptron
    bias = 0
    w = [0]*len(letras[0])
    alfa = 1
    cond_parada = True
    
    while cond_parada:
        for j in range(len(letras)):
            letra = letras[j]
            saida = saidas[j]
            yliq = bias
            for i in range(len(letra)):
                yliq += w[i]*letra[i]
            if yliq >= 0:
                y = 1
            else:
                y = -1
            if y != saida:
                for i in range(len(letra)):
                    w[i] += alfa*letra[i]*saida 
                bias += alfa*y
                cond_parada = False 
    return w,bias

def neuronio(w,bias,letra): #função para definir o neuronio
    yin = bias
    for j in range(len(letra)):
        yin += w[j]*letra[j]
    if yin >= 0:
        y = 1
    else:
        y = -1
    return y

if __name__ == "__main__":
    main()


        



