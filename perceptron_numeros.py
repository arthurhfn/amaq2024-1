def main():   
    x_0 = [-1,  1,  1,  1,  1,  1,  1, -1,
         1, -1, -1, -1, -1, -1,  -1,  1,
         1, -1, -1, -1, -1, -1,  -1,  1,
         1, -1, -1, -1, -1, -1,  -1,  1,
         1, -1, -1, -1, -1, -1,  -1,  1,
         1, -1, -1, -1, -1, -1,  -1,  1,
         1, -1, -1, -1, -1, -1,  -1,  1,
        -1,  1,  1,  1,  1,  1,  1, -1]

    x_1 = [-1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1,  1,  1,  1, -1, -1, -1,
        -1,  1, -1,  1,  1, -1, -1, -1,
         1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1, -1,  1,  1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1]

    x_2 = [-1,  1,  1,  1,  1,  1,  1, -1,
         1, -1, -1, -1, -1, -1,  1,  1,
        -1, -1, -1, -1, -1,  1,  1,  1,
        -1, -1, -1, -1,  1,  1,  1, -1,
        -1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1,  1,  1, -1, -1, -1, -1,
        -1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1]

    x_3 = [-1,  1,  1,  1,  1,  1,  1, -1,
         1, -1, -1, -1, -1, -1,  1,  1,
        -1, -1, -1, -1, -1,  1,  1, -1,
        -1,  1,  1,  1,  1,  1,  1, -1,
        -1, -1, -1, -1, -1, -1,  1,  1,
         1, -1, -1, -1, -1, -1,  1,  1,
        -1,  1,  -1,  -1,  -1,  -1,  1, -1,
         -1,  -1,  1,  1,  1,  1,  -1, -1]

    x_4 = [-1, -1, -1, -1,  1,  1, -1, -1,
        -1, -1, -1,  1,  1,  1, -1, -1,
        -1, -1,  1, -1,  1,  1, -1, -1,
        -1,  1, -1, -1,  1,  1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1,  1,  1, -1, -1,
        -1, -1, -1, -1,  1,  1, -1, -1,
        -1, -1, -1, -1,  1,  1, -1, -1]

    x_5 = [ 1,  1,  1,  1,  1,  1,  1,  1,
         1, -1, -1, -1, -1, -1, -1, -1,
         1,  -1,  -1,  -1,  -1,  -1,  -1, -1,
        1, 1, 1, 1, 1, 1,  1,  -1,
        -1, -1, -1, -1, -1, -1,  1,  1,
         1, -1, -1, -1, -1, -1,  -1,  1,
        -1,  1,  -1,  -1,  -1,  -1,  1, 1,
        -1, -1,  1,  1,  1,  1, -1, -1]

    x_6 = [-1, -1,  1,  1,  1,  1, -1, -1,
        -1,  1, -1, -1, -1, -1, -1, -1,
         1,  -1,  -1,  -1,  -1,  -1, -1, -1,
         1, 1, 1, 1, 1, 1,  1, -1,
         1, -1, -1, -1, -1, -1,  -1, 1,
         1, -1, -1, -1, -1, -1,  -1, 1,
        -1,  1,  -1,  -1,  -1,  -1,  1, -1,
        -1, -1, 1, 1,  1,  1, -1, -1]

    x_7 = [ 1,  1,  1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,  1,
        -1, -1, -1, -1, -1,  1,  1, -1,
        -1, -1, -1, -1,  1,  1, -1, -1,
        -1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1,  1,  1, -1, -1, -1, -1,
        -1,  1,  1, -1, -1, -1, -1, -1,
        -1,  1,  1, -1, -1, -1, -1, -1]

    x_8 = [-1, -1,  1,  1,  1,  1, -1, -1,
       -1,  1, -1, -1, -1, -1,  1, -1,
        1, -1, -1, -1, -1, -1,  -1,  1,
        -1, 1,  1,  1,  1,  1,  1,  -1,
        -1, 1, 1, 1, 1, 1,  1, -1,
        1, -1, -1, -1, -1, -1,  -1,  1,
       -1,  1, -1, -1, -1, -1,  1, -1,
       -1, -1,  1,  1,  1,  1, -1, -1]

    x_9 = [-1, -1, -1,  1,  1,  1, 1, 1,
        -1, -1, 1, -1, -1, -1,  -1,  1,
        -1, 1, -1, -1, -1, -1,  -1,  1,
        -1, -1, 1, -1, -1, -1,  -1, 1,
        -1, -1, -1, 1, 1, 1,  1,  1,
        -1,  -1,  -1, -1, -1, -1,  -1, 1,
       -1,  -1,  -1,  -1,  -1,  -1,  -1, 1,
       -1, -1,  1,  1,  1,  1, 1, 1]
    
    lista_numeros = [x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9]


    treino = calc_pesos(lista_numeros)
    resultado = rede_neuronio(treino,x_9)
    print(resultado)

'''    for k in range(10):
        for i in range(8):
            for j in range(8):
                print(f"{'.' if lista_numeros[k][8*i+j] == -1 else '#'}", end='')
            print()
        print()'''

def perceptron(numeros,saidas): #função perceptron
    bias = 0
    w = [0]*len(numeros[0])
    alfa = 1
    
    while True:
        cond_parada = False
        for j in range(len(numeros)):
            numero = numeros[j]
            saida = saidas[j]
            yliq = bias
            for i in range(len(numero)):
                yliq += w[i]*numero[i]
            if yliq >= 0:
                y = 1
            else:
                y = -1
            if y != saida:
                for i in range(len(numero)):
                    w[i] += alfa*numero[i]*saida 
                bias += alfa*saida
                cond_parada = True
        if not cond_parada:
            break
    return w,bias

def neuronio(w,bias,numero): #função para definir o neuronio
    yin = bias
    for j in range(len(numero)):
        yin += w[j]*numero[j]
    if yin >= 0:
        y = 1
    else:
        y = -1
    return y

def calc_pesos(numeros):
    pesos = []
    for j in range(len(numeros)):
        saidas = [-1]*10
        saidas[j] = 1

        pesos.append(perceptron(numeros,saidas)) 
        
    return pesos

def rede_neuronio(pesos,numero):
    reconhece = []
    for j in range(len(pesos)):
        reconhece.append(neuronio(pesos[j][0],pesos[j][1],numero)) 

    return reconhece
        

if __name__ == "__main__":
    main()


        



