using LinearAlgebra, Plots, Printf, BenchmarkTools, SparseArrays
gr(size=(600,400))

"""
Função para tornar entradas como "1.2314e-16" iguais a "0".
"""
function ehzero(A)
    for i = 1 : size(A,1)
        for j = 1 : size(A,2)
            if abs(A[i,j]) < 1e-15
                A[i,j] = 0
            end
        end
    end
end

############### qr por gram schmidt ###############
"""
A função QR_GS calcula a decomposição QR de uma matriz A por meio
do método de decomposição QR de Gram-Schmidt.
"""
function QR_GS(A, ntol = 1e-12)
    n = size(A,1)
    m = size(A,2)
    v = A[:,1]
    norm_v = dot(v,v)^(1/2)
    if norm_v == 0
        println("norm_v = $norm_v")
        error("A decomposição QR não pode ser feita para uma matriz A que tem colunas LD")
    elseif norm_v < ntol
        println("norm_v = $norm_v")
        error("A decomposição QR por Gram-Schmidt não é adequada para casos em que a norma de uma das colunas de A é muito próxima de zero.\n
        Ao invés disso, utilize a decomposição QR por rotação.")
    end
    Q = v.*(1/norm_v)
    for i = 2:m
        u = A[:,i]
        somatorio = zeros(n,1)
        for j = 1:i-1
            somatorio += dot(Q[:,j],u) .* Q[:,j]
        end
        v = u - somatorio
        norm_v = dot(v,v)^(1/2)
        if norm_v == 0
            println("norm_v = $norm_v")
            error("A decomposição QR não pode ser feita para uma matriz A que tem colunas LD")
        elseif norm_v < ntol
            println("norm_v = $norm_v")
            error("A decomposição QR por Gram-Schmidt não é adequada para casos em que as colunas estão 'muito próximas de serem LD' ")
        end
        v = v.*(1/norm_v)
        Q = [Q v]
    end
    R = zeros(m,m)
    for i = 1:m
        for j = i:m
            R[i,j] = dot(Q[:,i],A[:,j])
        end
    end
    return Q, R
end


############### qr por rotação ###############


"""
A função QR_rotacao calcula a decomposição QR de uma matriz A por meio
do método de decomposição QR por rotação.
"""
function QR_rotacao(A, etol = 1e-14)
    n, m = size(A,1), size(A,2)
    if A == zeros(n,m)
        error("A deve ser não nula")
    end
    R = copy(A)
    a = k = 0
    if n == m
        a = m - 1
    else
        a = m
    end
    j, i = 1, 2
    Q = zeros(n,n) + I
    while j <= a
        while i <= n
            if abs(R[i,j]) > etol
                pow = (R[j,j])^2
                norma = sqrt(pow + (R[i,j])^2)
                c = R[j,j] / norma
                s = R[i,j] / norma
                R[j,:], R[i,:] = (c * R[j,:]) + (s * R[i,:]), ((-s) * R[j,:]) + (c * R[i,:])
                Q[j,:], Q[i,:] = (c *  Q[j,:]) + (s * Q[i,:]), ((-s) * Q[j,:]) + (c * Q[i,:])
            end
            i += 1
        end
        i = 3 + k
        j = j + 1
        k = k + 1
    end
    Q = Q'
    ehzero(Q)
    ehzero(R)
    return Q, R
end


################# resolver sistema triangular superior #################

"""
A função sist tri sup resolve sistemas da forma A*x = b, em que A é uma matriz triangular superior.
Isso economiza muita memória e alocações, bem mais vantajoso do que usar a barra invertida.
"""
function sist_tri_sup(A, b)
    n = length(b)
    x = zeros(n)
    s = 0
    for i = n : -1 : 1
        if abs(A[i,i]) < 1e-12
            error()
        end
        s = b[i]
        for j = (i+1):n
            s = s - A[i,j]*x[j]
        end
        x[i] = s / A[i,i]
    end
    return x
end

############### quadrados minimos ###############

"""
A função quad min GS calcula a solução x do sistema Ax = b, sendo
A = QR pela fatoração de Gram-Schmidt. Ela recebe como parâmetros
a matriz "A" e o vetor "b".
"""
function quad_min_GS(A, b)
    if size(A,1) != size(b,1)
        error("A e b devem ter mesmo número de linhas para que seja possível resolver o sistema A*x=b")
    end
    Q, R = QR_GS(A)
    x = sist_tri_sup(R, Q' * b )
    return x
end


"""
A função quad min rotacao calcula a solução x do sistema Ax = b, sendo
A = QR pela fatoração por rotação. Ela recebe como parâmetros
a matriz "A" e o vetor "b".
"""
function quad_min_rotacao(A, b)
    if size(A,1) != size(b,1)
        error("A e b devem ter mesmo número de linhas para que seja possível resolver o sistema A*x=b")
    end
    Q, R = QR_rotacao(A)
    m = size(R, 2)
    R_1 = R[1:m, :] #nova matriz R, agora quadrada
    for i = 1:size(R_1,2)
        if R_1[i,i] == 0
            error("A decomposição QR não pode ser feita para uma matriz A que tem colunas LD")
        end
    end
    y = Q' * b
    Q_1 = y[1:m, :]
    x = sist_tri_sup(R_1,Q_1)
    return x
end

############### gráfico #############

"""
A função grafico quad min transforma pontos dados em uma matriz A,
que gera como solução os coeficientes "a" e "b" de uma reta y = ax + b,
sendo esta a reta que "passa mais perto" dos pontos dados.

Na entrada da função, as coordenadas dos pontos devem ser dadas como no exemplo abaixo:\n
Considere os pontos P1 = (x1,y1) e P2 = (x2,y2)\n
Então x = [x1; x2] e y = [y1; y2]
"""
function grafico_quad_min(x::Vector, y::Vector)
    n = length(x)
    A = [x ones(n)]
    reta = quad_min_rotacao(A,y)
    reta2 = quad_min_GS(A,y)
    scatter(x, y, leg=false, c=:lightblue, ms = 5)
    if reta[2] < 0
        b = abs(reta[2])
        b2 = abs(reta2[2])
        png(plot!(x-> reta[1]*x + reta[2], c=:red, lw=2, title="Reta: y = $(reta[1])*x - $b"),"Grafico por rotacao")
        png(plot!(x-> reta2[1]*x + reta2[2], c=:red, lw=2, title="Reta: y = $(reta2[1])*x - $b2"),"Grafico por GS")
    else
        png(plot!(x-> reta[1]*x + reta[2], c=:red, lw=2, title="Reta: y = $(reta[1])*x + $(reta[2])"),"Grafico por rotacao")
        png(plot!(x-> reta2[1]*x + reta2[2], c=:red, lw=2, title="Reta: y = $(reta2[1])*x + $(reta2[2])"),"Grafico por GS")
    end
end

################ tabela ##################

function gerar_tabela(A,b)
    erro_GS = abs(norm(b)-norm(A*quad_min_GS(A,b)))
    erro_rot = abs(norm(b)-norm(A*quad_min_rotacao(A,b)))
    tempo_GS = @belapsed quad_min_GS(A,b)
    tempo_rot = @belapsed quad_min_rotacao(A,b)
    @printf("%-9s | %10s | %10s\n", "  ", "Gram-Schmidt", "Rotação")
    @printf("%-9s | %10s | %10s\n", "-"^9, "-"^12, "-"^13)
    @printf("%-9s | %-12s | %10s\n", "Erro de b" , @sprintf("%10g",erro_GS) , @sprintf("%10g",erro_rot))
    @printf("%-9s | %10s | %10s\n", "-"^9, "-"^12, "-"^13)
    @printf("%-9s | %.10f | %.10f\n", "Tempo (s)" , tempo_GS , tempo_rot)
    @printf("%-9s | %10s | %10s\n", "-"^9, "-"^12, "-"^13)
end

################ main ################

function main()
    clearconsole()
    Q, R = QR_GS(A)
    println("Após fazer a decomposição QR por Gram-Schmidt, temos que:", "\n")
    println("Q é a matriz abaixo")
    println("---------------------")
    display(Q)
    println("---------------------")
    println("R é a matriz abaixo")
    println("---------------------")
    display(R)
    println("---------------------")
    Q, R = QR_rotacao(A)
    println("Após fazer a decomposição QR por rotação, temos que:", "\n")
    println("Q é a matriz abaixo")
    println("---------------------")
    display(Q)
    println("---------------------")
    println("R é a matriz abaixo")
    println("---------------------")
    display(R)
    println("---------------------")
    x = quad_min_GS(A,b)
    println("A solução mais próxima para resolver o sistema A*x = b por Gram-Schmidt é x = $x\nx é o vetor mostrado abaixo")
    println("---------------------")
    display(x)
    println("---------------------")
    y = quad_min_rotacao(A,b)
    println("A solução mais próxima para resolver o sistema A*x = b por rotação é x = $y\nx é o vetor mostrado abaixo")
    println("---------------------")
    display(y)
    println("---------------------")
    println("Abaixo está uma tabela, comparando:\nA precisão do resultado A*x, sendo x a solução encontrada, e b.\nO tempo levado para gerar as soluções")
    gerar_tabela(A,b)
    if size(A,2) == 2
        if A[:,2] == ones(size(A,1))
            x = A[:,1]
            println("A matriz A é um caso particular de aproximação linear para quadrados mínimos.\n
Sua coluna 1 representa as coordenadas x's de pontos no plano R2, e o vetor b representa as coordenadas y's.\n
No gráfico plotado, os pontos azuis são esses mencionados e a reta em vermelho é a reta que passa mais perto desses pontos.")
            grafico_quad_min(x,b)
        end
    end
end

############### testes ###############

#Exemplo 1
A = [0.0 1.0; 1.0 1.0; 2.0 1.0; 3.0 1.0; 4.0 1.0]
b = [-5; 0; 5; 12; 24]

#Exemplo 2
#A = [1.0 0.0 -1.0; 2.0 1.0 -2.0; 1.0 1.0 0.0]
#b = [6.0; 0.0; 9.0]

#Exemplo 3
#A = [4.0 1.0; 6.0 1.0; 8.0 1.0; 10.0 1.0; 12.0 1.0; 14.0 1.0]
#b = [7.0; 9.0; 4.0; 6.0; 2.0; 1.0 ]

#Exemplo com vetores LD
#A = [1 4; 2 8]
#b = [8; 16]

#Exemplo de matriz esparsa (não-densa)
#A = I - Matrix(sprand(10,10, 0.1))
#b = rand(10,1)

#Exemplo de matriz densa (não-esparsa)
#A = rand(10,10)
#b = rand(10,1)

#Exemplo de matriz "densa e esparsa" (100x100 com 50% das entradas sendo 0)
#A = Matrix(sprand(100,100,0.5))
#b = rand(100,1)

main()
