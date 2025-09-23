#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    # Matrizes
    Mt = None
    Mr = None
    Ms = None

    Mt_c = None
    Mr_c = None
    
    Mp = None

    # Pilha
    stack = []
    last_matrix = np.identity(4)

    # Supersampling (anti-aliasing)
    supersampling_active = False
    supersampling_size = 2
    if not supersampling_active:
        supersampling_size = 1

    # Camera
    cam_direction = (0, 0, -1)

    # Iluminação (Headlight e AmbientLight)
    hasLight = False
    headlight = False
    light_intensity = 0
    light_ambientIntensity = 0.2
    ambientIntensity = 0.0
    light_color = (0, 0, 0)
    light_direction = (0, 0, -1)
    expoente_reflex_espec = 0.2

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        len_lista = len(point)
        r, g, b = colors["emissiveColor"]
        for i in range(0, len_lista, 2):
            pos_x = int(point[i])
            pos_y = int(point[i+1])
            gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [r*255, g*255, b*255])

        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # print("\n\nPolyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        lista_pontos = []
        r, g, b = colors["emissiveColor"]
        for i in range(0, len(lineSegments), 2):
            lista_pontos.append([int(lineSegments[i]), int(lineSegments[i+1])])

        for i in range(len(lista_pontos) - 1):
            p0 = lista_pontos[i]
            p1 = lista_pontos[(i+1) % len(lista_pontos)]

            # Coeficiente angular : (y1 - y0)/(x1 - x0)
            y1_y0 = (p1[1] - p0[1])
            x1_x0 = (p1[0] - p0[0])
            if x1_x0 == 0:
                coef_ang = y1_y0
            else:
                coef_ang = y1_y0 / x1_x0

            if abs(coef_ang) < 1: # se x cresce mais que y (percorrer x e preencher y)

                if p0[0] <= p1[0]:
                    maior = p1
                    menor = p0
                else:
                    maior = p0
                    menor = p1
                v = menor[1]

                if p1[0] - p0[0] == 0:
                    coef_ang = p1[1] - p0[1]
                else:
                    coef_ang = (p1[1] - p0[1])/(p1[0] - p0[0])

                for u in range(menor[0], maior[0] + 1):
                    # print(f"{u}, {v}")
                    try:
                        gpu.GPU.draw_pixel([u, round(v)], gpu.GPU.RGB8, [r*255, g*255, b*255])
                    except:
                        pass
                    v += coef_ang
                
            else: # se y cresce mais que x (percorrer y e preencher x)

                if p0[1] <= p1[1]:
                    maior = p1
                    menor = p0
                else:
                    maior = p0
                    menor = p1
                u = menor[0]

                if p1[1] - p0[1] == 0:
                    coef_ang = p1[0] - p0[0]
                else:
                    coef_ang = (p1[0] - p0[0])/(p1[1] - p0[1])

                for v in range(menor[1], maior[1] + 1):
                    # print(f"== {u}, {v}")
                    try:
                        gpu.GPU.draw_pixel([round(u), v], gpu.GPU.RGB8, [r*255, g*255, b*255])
                    except:
                        pass
                    u += coef_ang
            

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        # print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # ISSO NAO VAI FUNCIONAR SE O CENTRO FOR DIFERENTE DE (0, 0)!!!!!!!!!!

        r, g, b = colors["emissiveColor"]
        pontos_por_lado = 4 # quantos pontos colocar por "lado" do circulo
        lista_pontos = [] # lista dos pontos que realmente serao desenhados
        pontos_obrigatorios = [[int(radius), 0], [0, int(radius)], [-int(radius), 0], [0, -int(radius)]] # pontos chave
        for i in range(len(pontos_obrigatorios)):
            x0, y0 = pontos_obrigatorios[i]
            x1, y1 = pontos_obrigatorios[(i+1) % len(pontos_obrigatorios)]
            x, y = x0, y0
            mult = 1
            if x < 0 or y < 0:
                mult = -1
            for _ in range(pontos_por_lado):
                lista_pontos.append((x, y))
                x = int(x + (x1 - x0)/pontos_por_lado)
                y = int(mult * math.sqrt(int(radius)**2 - x**2))

        for i in range(len(lista_pontos) - 1):
            p0 = lista_pontos[i]
            p1 = lista_pontos[(i+1) % len(lista_pontos)]

            # Coeficiente angular : (y1 - y0)/(x1 - x0)
            y1_y0 = (p1[1] - p0[1])
            x1_x0 = (p1[0] - p0[0])
            if x1_x0 == 0:
                coef_ang = y1_y0
            else:
                coef_ang = y1_y0 / x1_x0

            if abs(coef_ang) < 1: # se x cresce mais que y (percorrer x e preencher y)

                if p0[0] <= p1[0]:
                    maior = p1
                    menor = p0
                else:
                    maior = p0
                    menor = p1
                v = menor[1]

                if p1[0] - p0[0] == 0:
                    coef_ang = p1[1] - p0[1]
                else:
                    coef_ang = (p1[1] - p0[1])/(p1[0] - p0[0])

                for u in range(menor[0], maior[0] + 1):
                    # print(f"{u}, {v}")
                    try:
                        gpu.GPU.draw_pixel([u, round(v)], gpu.GPU.RGB8, [r*255, g*255, b*255])
                    except:
                        pass
                    v += coef_ang
                
            else: # se y cresce mais que x (percorrer y e preencher x)

                if p0[1] <= p1[1]:
                    maior = p1
                    menor = p0
                else:
                    maior = p0
                    menor = p1
                u = menor[0]

                if p1[1] - p0[1] == 0:
                    coef_ang = p1[0] - p0[0]
                else:
                    coef_ang = (p1[0] - p0[0])/(p1[1] - p0[1])

                for v in range(menor[1], maior[1] + 1):
                    # print(f"== {u}, {v}")
                    try:
                        gpu.GPU.draw_pixel([round(u), v], gpu.GPU.RGB8, [r*255, g*255, b*255])
                    except:
                        pass
                    u += coef_ang

    def draw_line(x0, y0, x1, y1, r, g, b):
        y1_y0 = (y1 - y0)
        x1_x0 = (x1 - x0)

        if x1_x0 == 0:
            coef_ang = y1_y0
        else:
            coef_ang = y1_y0 / x1_x0

        if abs(coef_ang) < 1: # se x cresce mais que y (percorrer x e preencher y)

            if x0 <= x1:
                maior = x1
                menor = x0
                v = y0
            else:
                maior = x0
                menor = x1
                v = y1

            if x1 - x0 == 0:
                coef_ang = y1 - y0
            else:
                coef_ang = (y1 - y0)/(x1 - x0)

            for u in range(int(menor), round(maior) + 1):
                gpu.GPU.draw_pixel([u, round(v)], gpu.GPU.RGB8, [r*255, g*255, b*255])
                v += coef_ang
            
        else: # se y cresce mais que x (percorrer y e preencher x)

            if y0 <= y1:
                maior = y1
                menor = y0
                u = x0
            else:
                maior = y0
                menor = y1
                u = x1

            if y1 - y0 == 0:
                coef_ang = x1 - x0
            else:
                coef_ang = (x1 - x0)/(y1 - y0)

            for v in range(int(menor), round(maior) + 1):
                gpu.GPU.draw_pixel([round(u), v], gpu.GPU.RGB8, [r*255, g*255, b*255])
                u += coef_ang

    # Calculo de alpha, beta e gama para Coord. Baricêntricas (Aula 8)
    def calcula_alpha_beta_gama(vertices, x, y):
        # Pega os pontos 
        (xA, yA, _, _) = vertices[0]
        (xB, yB, _, _) = vertices[1]
        (xC, yC, _, _) = vertices[2]

        # Calcula Alpha
        num_alpha = -1 * (x  - xB)*(yC - yB) + (y  - yB)*(xC - xB)
        den_alpha = -1 * (xA - xB)*(yC - yB) + (yA - yB)*(xC - xB)
        alpha = num_alpha/den_alpha

        # Calcula Beta
        num_beta = -1 * (x  - xC)*(yA - yC) + (y  - yC)*(xA - xC)
        den_beta = -1 * (xB - xC)*(yA - yC) + (yB - yC)*(xA - xC)
        beta = num_beta/den_beta

        # Calcula Gama
        gama = 1 - alpha - beta

        return (alpha, beta, gama)
    
    def calcula_uv(textCoords, pesosCoordBar, listaW):
        # Pesos 
        (alpha, beta, gama) = pesosCoordBar

        # Valores W
        (w0, w1, w2) = listaW

        # Coord. Texturas (Aula 09)
        u0, v0 = textCoords[0]
        u1, v1 = textCoords[1]
        u2, v2 = textCoords[2]

        # Calcula coord. em u
        u = alpha*(u0/w0) + beta*(u1/w1) + gama*(u2/w2)  
        u /= alpha*(1/w0) + beta*(1/w1) + gama*(1/w2)  
        
        # Calcula coord. em v
        v = alpha*(v0/w0) + beta*(v1/w1) + gama*(v2/w2)
        v /= alpha*(1/w0) + beta*(1/w1) + gama*(1/w2)
        
        return u, v
    
    # Devolve o MipMap de textura com uma nova resolucao com base no D
    # Ex.: 32x32, 16x16, 8x8, etc.
    def img_MipMaps(img, imgShape):
        mipMaps = []

        w, h, z = imgShape
        while w > 1 and h > 1:

            resized = np.zeros((w//2, h//2, z))
            for u in range(0, w, 2):
                for v in range(0, h, 2):
                    # Pega os texels (texture pixels)
                    texel0 = img[int(u), int(v)]
                    texel1 = img[int(u)+1, int(v)]
                    texel2 = img[int(u), int(v)+1]
                    texel3 = img[int(u)+1, int(v)+1]

                    # Calcula a media RGB entre os 4 pontos
                    r = (float(texel0[0]) + float(texel1[0]) + float(texel2[0]) + float(texel3[0]))//4
                    g = (float(texel0[1]) + float(texel1[1]) + float(texel2[1]) + float(texel3[1]))//4
                    b = (float(texel0[2]) + float(texel1[2]) + float(texel2[2]) + float(texel3[2]))//4

                    resized[int(u)//2, int(v)//2] = [r, g, b, 255]

            mipMaps.append(resized)
            w //= 2
            h //= 2

        return mipMaps
    
    def calcula_normais(lista_pontos, indices):
        # Passa para np.array (fica mais facil)
        lista_pontos = [np.array(v[:3], dtype=float) for v in lista_pontos]
        normais = [np.array([0.0, 0.0, 0.0]) for _ in lista_pontos]
        
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
            p0, p1, p2 = lista_pontos[i0], lista_pontos[i1], lista_pontos[i2]

            v0 = p1 - p0
            v1 = p2 - p0
            n = np.cross(v0, v1)

            if np.linalg.norm(n) > 0:
                n = n / np.linalg.norm(n)

            normais[i0] += n
            normais[i1] += n
            normais[i2] += n

        # Normaliza cada normal acumulada
        normais = [n/np.linalg.norm(n) if np.linalg.norm(n) > 0 else n
                        for n in normais]

        return normais

    def draw_triangle(lista_pontos, r, g, b, colorPerVertex=False, vertexColors=None, transparencia=1, 
        hasTexture=False, textCoords=None, textShape=(0,0), textImg=None, diffuseColor=(1,1,1), specularColor=(1,1,1),
        shininess = 1, emissiveColor = [0, 0, 0], vertexNormals=None
    ):
        def inside(triangle, x, y):
            # print()
            for i in range(len(triangle)):
                p0 = triangle[i]
                p1 = triangle[(i+1) % len(triangle)]
                L = (p1[1] - p0[1])*x - (p1[0] - p0[0])*y + p0[1]*(p1[0] - p0[0]) - p0[0]*(p1[1] - p0[1])
                # if x == 43.5 and y == 11.5:
                #     print(x, y, L, triangle)
                if L < 0:
                    return 0
            return 1
        
        # Caso use textura (resize com base em D)
        hasMipMap = 0
        
        # Coordenadas do Baricentro (soma no 'for' e dps faz a media aritmetica)
        xG = 0
        yG = 0
        zG = 0

        # Desenha os vertices do triangulo
        for i in range(3):
            x = lista_pontos[i][0]       # x do vertice
            y = lista_pontos[i][1]       # y do vertice
            z = lista_pontos[i][2]       # z do vertice

            # Adiciona para o Baricentro
            xG += x
            yG += y
            zG += z
            
            if colorPerVertex:
                r_pnt = vertexColors[i][0]
                g_pnt = vertexColors[i][1]
                b_pnt = vertexColors[i][2]
                
                gpu.GPU.draw_pixel([round(x) * GL.supersampling_size, round(y) * GL.supersampling_size], gpu.GPU.RGB8, [r_pnt*255, g_pnt*255, b_pnt*255])
            else:
                gpu.GPU.draw_pixel([round(x) * GL.supersampling_size, round(y) * GL.supersampling_size], gpu.GPU.RGB8, [r*255, g*255, b*255])

        # for i in range(len(lista_pontos)):
            # GL.draw_line(lista_pontos[i][0], lista_pontos[i][1], lista_pontos[(i+1)%3][0], lista_pontos[(i+1)%3][1], r, g, b)
        
        if colorPerVertex:
            # Termina a media aritmetica do Baricentro
            xG = xG/3
            yG = yG/3 
            zG = zG/3

        min_x = min(lista_pontos[0][0], lista_pontos[1][0], lista_pontos[2][0])
        max_x = max(lista_pontos[0][0], lista_pontos[1][0], lista_pontos[2][0])

        min_y = min(lista_pontos[0][1], lista_pontos[1][1], lista_pontos[2][1])
        max_y = max(lista_pontos[0][1], lista_pontos[1][1], lista_pontos[2][1])

        plus_x, plus_y = [0.5], [0.5]
        if GL.supersampling_active:
            plus_x = [0.25, 0.75]
            plus_y = [0.25, 0.75]

        # Se navigationInfo (headlight = True)
        if GL.hasLight:
            vertex_colors = []
            for i, p in enumerate(lista_pontos):  
                # Vetor Normal
                N = vertexNormals[i]
                N /= np.linalg.norm(N)

                # Vetor L (contrario a direcao da luz)
                if GL.headlight:
                    L = -np.array(GL.light_direction)
                else:
                    L = np.array(GL.light_direction)
                L = L / np.linalg.norm(L)

                # Vetor v (contrario a camera)
                v = -np.array(GL.cam_direction)
                v = v / np.linalg.norm(v)

                # Produto escalar
                NL = np.dot(N, L)

                # Bissetriz
                Lv_Lv = (L + v) / np.linalg.norm(L + v)
                NLv_Lv = np.dot(N, Lv_Lv)

                # Calcula a cor do pixel
                Irgb = []
                for i in range(3):
                    ambient_i = diffuseColor[i]*max(0.2,GL.ambientIntensity)
                    diffuse_i = diffuseColor[i]*GL.light_intensity*NL
                    specular_i = 0.0
                    if NL > 0:
                        specular_i = specularColor[i]*GL.light_intensity*NLv_Lv**(shininess*128)
                    # print('_is: ', ambient_i, diffuse_i, specular_i)

                    soma_i = GL.light_color[i] * (ambient_i+diffuse_i+specular_i)
                    # print('soma_i: ', soma_i)
                    Irgb.append(emissiveColor[i] + soma_i)
                # Salva as cores do ponto
                vertex_colors.append(Irgb)


        for w in range(max(int(min_x), 0), min(round(max_x) + 1, GL.width)):
            for h in range(max(int(min_y), 0), min(round(max_y) + 1, GL.height)):
                for pl_x in plus_x:
                    for pl_y in plus_y:
                        if inside(lista_pontos, w + pl_x, h + pl_y):
                            # Fórmulas das Coordenadas Baricêntricas
                            (alpha, beta, gama) = GL.calcula_alpha_beta_gama(lista_pontos, w, h)

                            # Calculando Z_buffer do ponto
                            z0, z1, z2 = abs(lista_pontos[0][2]), abs(lista_pontos[1][2]), abs(lista_pontos[2][2])
                            z_buff = 1/(alpha/z0 + beta/z1 + gama/z2)

                            # Pega os valores W (depois de Projection - 3D to 2D)
                            w0 = lista_pontos[0][3]
                            w1 = lista_pontos[1][3]
                            w2 = lista_pontos[2][3]
                            listaW = (w0, w1, w2)

                            if GL.hasLight:
                                # Z para interpolação de cores
                                z = 1/(alpha/w0 + beta/w1 + gama/w2)

                                # Pega as cores dos vertices
                                (r_v0, g_v0, b_v0) = vertex_colors[0]
                                (r_v1, g_v1, b_v1) = vertex_colors[1]
                                (r_v2, g_v2, b_v2) = vertex_colors[2]

                                r = max(0, min(1, z * (alpha*r_v0/w0 + beta*r_v1/w1 + gama*r_v2/w2)))
                                g = max(0, min(1, z * (alpha*g_v0/w0 + beta*g_v1/w1 + gama*g_v2/w2)))
                                b = max(0, min(1, z * (alpha*b_v0/w0 + beta*b_v1/w1 + gama*b_v2/w2)))
                            
                            if hasTexture:
                                pesos = (alpha, beta, gama)

                                # Coordenadas uv
                                u, v = GL.calcula_uv(textCoords, pesos, listaW)

                                # Coordenadas uv para vizinho (direita)
                                x_right, y_right = w+1, h
                                (alpha_right, beta_right, gama_right) = GL.calcula_alpha_beta_gama(lista_pontos, x_right, y_right)
                                pesos_right = (alpha_right, beta_right, gama_right)

                                u_right, v_right = GL.calcula_uv(textCoords, pesos_right, listaW)

                                # Coordenadas uv para vizinho (abaixo)
                                x_down, y_down = w, h+1
                                (alpha_down, beta_down, gama_down) = GL.calcula_alpha_beta_gama(lista_pontos, x_down, y_down)
                                pesos_down = (alpha_down, beta_down, gama_down)

                                u_down, v_down = GL.calcula_uv(textCoords, pesos_down, listaW)

                                # Derivadas parciais
                                dudx = (u_right - u)/(x_right - w)
                                dudy = (u_down - u)/(y_down - h)
                                dvdx = (v_right - v)/(x_right - w)
                                dvdy = (v_down - v)/(y_down - h)

                                # Calculo do L
                                L = max( (dudx**2 + dvdx**2)**(1/2), (dudy**2 + dvdy**2)**(1/2) )

                                # "Transforma" a imagem para resolucao de acordo com D, se ja n tiver feito
                                if hasMipMap == 0:
                                    mipMaps = GL.img_MipMaps(textImg, textShape)
                                    hasMipMap = 1

                                # So para evitar valores negativos
                                D = max(0, min(len(mipMaps), int(math.log2(L))))

                                # Pega o ponto na textura, apos o resize 
                                mipMap_h, mipMap_w, _ = mipMaps[D].shape
                                tex_x = int(u * (mipMap_w  - 1))
                                tex_y = int((1-v) * (mipMap_h - 1))     # Inverte v, pois no grafico da aula 'v' tem sentido oposto a y na tela

                                rgb = mipMaps[D][tex_x][tex_y]
                                r_pixel = rgb[0]
                                g_pixel = rgb[1]
                                b_pixel = rgb[2]

                                gpu.GPU.draw_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.RGB8, [int(r_pixel), int(g_pixel), int(b_pixel)]) 
                            elif colorPerVertex:

                                # Z para interpolação de cores
                                z = 1/(alpha/w0 + beta/w1 + gama/w2)

                                # Pega as cores dos vertices
                                (r_v0, g_v0, b_v0) = vertexColors[0]
                                (r_v1, g_v1, b_v1) = vertexColors[1]
                                (r_v2, g_v2, b_v2) = vertexColors[2]
            
                                # Calculo de cor do ponto, de acordo com seu alpha beta gama e z 
                                r_pixel = max(0, min(1, z * (alpha*r_v0/w0 + beta*r_v1/w1 + gama*r_v2/w2)))
                                g_pixel = max(0, min(1, z * (alpha*g_v0/w0 + beta*g_v1/w1 + gama*g_v2/w2)))
                                b_pixel = max(0, min(1, z * (alpha*b_v0/w0 + beta*b_v1/w1 + gama*b_v2/w2)))

                                # Atualizando cor apenas se z_buff for menor (mais na frente)
                                if z_buff < gpu.GPU.read_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.DEPTH_COMPONENT32F)[0]:
                                    gpu.GPU.draw_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.DEPTH_COMPONENT32F, [z_buff])
                                    gpu.GPU.draw_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.RGB8, [int(r_pixel*255), int(g_pixel*255), int(b_pixel*255)]) 
                            else:
                                # Atualizando cor apenas se z_buff for menor (mais na frente)
                                if z_buff < gpu.GPU.read_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.DEPTH_COMPONENT32F)[0]:
                                    r_final = r
                                    g_final = g
                                    b_final = b

                                    # Se for transparente
                                    if transparencia < 1:
                                        # Pegando cor antiga (e normalizando pra intervalo [0, 1]
                                        old_r, old_g, old_b = gpu.GPU.read_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.RGB8)
                                        old_r *= transparencia / 255
                                        old_g *= transparencia / 255
                                        old_b *= transparencia / 255

                                        new_r = r * (1 - transparencia)
                                        new_g = g * (1 - transparencia)
                                        new_b = b * (1 - transparencia)

                                        r_final = min(1, old_r + new_r)
                                        g_final = min(1, old_g + new_g)
                                        b_final = min(1, old_b + new_b)

                                    gpu.GPU.draw_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.DEPTH_COMPONENT32F, [z_buff])
                                    gpu.GPU.draw_pixel([GL.supersampling_size * w + round(pl_x), GL.supersampling_size * h + round(pl_y)], gpu.GPU.RGB8, [int(r_final*255), int(g_final*255), int(b_final*255)])        

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        # print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        # print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores

        lista_pontos_float = [] # lista com os valores em float para melhorar a conta da funcao inside (isso fez diferenca)
        r, g, b = colors["emissiveColor"]
        for i in range(0, len(vertices), 2):
            # lista_pontos.append([int(vertices[i]), int(vertices[i+1])])
            lista_pontos_float.append([vertices[i], vertices[i+1], 0.5, 1])
        # print(lista_pontos_float)
        for i in range(0, len(lista_pontos_float), 3):
            # print(lista_pontos_float[i:(i+3)])
            GL.draw_triangle(lista_pontos_float[i:(i+3)], r, g, b)

    def transform_3Dto2D(x_3d, y_3d, z_3d):
        M = GL.last_matrix

        # Object-World
        OW = M @ np.array([[x_3d], [y_3d], [z_3d], [1]])

        # World-View
        # MR @ MT <- Matriz Look At (Aula 6) (Camera)
        WV = GL.Mr_c @ GL.Mt_c @ OW

        # View-Projection
        VP = GL.Mp @ WV

        w = VP[3]
        x_w = VP[0] / w
        y_w = VP[1] / w
        z_w = VP[2] / w

        # Matrriz de Transformações para Tela
        M_screen = np.array([
            [GL.width/2, 0, 0, GL.width/2],
            [0, -GL.height/2, 0, GL.height/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        [x], [y], [z], _ = M_screen @ np.array([[x_w[0]], [y_w[0]], [z_w[0]], [1]])

        # print(z)
        # if z < gpu.GPU.read_pixel([round(x), round(y)], gpu.GPU.DEPTH_COMPONENT32F)[0]:
        #     gpu.GPU.draw_pixel([round(x), round(y)], gpu.GPU.DEPTH_COMPONENT32F, [z])

        # print(gpu.GPU.frame_buffer[x][y])
        # if z < gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F):
        # if z < gpu.GPU.frame_buffer[x][y]:
        #     gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)
            # gpu.GPU.
            # gpu.GPU.frame_buffer[x][y] = 0

        return (x, y, z, w)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleSet : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("TriangleSet : colors = {0}".format(colors)) # imprime no terminal as cores

        transparency = colors["transparency"]
        r, g, b = colors["emissiveColor"]

        # Caracteristicas do material
        diffuseColor = colors["diffuseColor"]
        specularColor = colors["specularColor"]
        shininess = colors["shininess"]
        
        lista_pontos = []
        vertices = []       # Sem transformar (Iluminacao)
        for i in range(0, len(point), 3):
            vertices.append(np.array([point[i], point[i + 1], point[i + 2]]))
            lista_pontos.append(GL.transform_3Dto2D(point[i], point[i + 1], point[i + 2]))

        # Índices diretos (cada 3 pontos = um triângulo)
        indices = list(range(len(lista_pontos)))

        # Calcula as normais por vertice (Iluminacao)
        normais = GL.calcula_normais(vertices, indices)
            
        for i in range(0, len(lista_pontos), 3):
            GL.draw_triangle(lista_pontos[i:(i+3)], r, g, b, transparencia=transparency, diffuseColor=diffuseColor, specularColor=specularColor, shininess=shininess,
                            vertexNormals=normais[i:(i+3)])

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Viewpoint : ", end='')
        # print("position = {0} ".format(position), end='')
        # print("orientation = {0} ".format(orientation), end='')
        # print("fieldOfView = {0} ".format(fieldOfView))

        # Ex.: Viewpoint : position = [0.0, 0.0, -5.0] orientation = [0.0, -1.0, 0.0, 3.1415] fieldOfView = 0.7853981633974483 
        # Posicao da câmera
        Cx, Cy, Cz = position[0], position[1], position[2]
        GL.Mt_c = np.array([
            [1, 0, 0, -Cx],
            [0, 1, 0, -Cy],
            [0, 0, 1, -Cz],
            [0, 0, 0, 1 ]
        ])
        
        # Orientação da câmera
        qi = orientation[0] * math.sin(orientation[3]/2)
        qj = orientation[1] * math.sin(orientation[3]/2)
        qk = orientation[2] * math.sin(orientation[3]/2)
        qr = math.cos(orientation[3]/2)
        GL.Mr_c = np.array([
            [1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0],
            [2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr), 0],
            [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2), 0],
            [0, 0, 0, 1]
        ])

        fovy = 2 * math.atan(math.tan(fieldOfView/2) * GL.height/(np.sqrt(GL.height**2 + GL.width**2)))
        top = GL.near * math.tan(fovy)
        right = top * GL.width/GL.height

        # Matriz de perspectiva 
        GL.Mp = np.array([
            [GL.near/right, 0, 0, 0],
            [0, GL.near/top, 0, 0],
            [0, 0, - (GL.far + GL.near)/(GL.far - GL.near), -2 * GL.far * GL.near / (GL.far - GL.near)],
            [0, 0, -1, 0]
        ])

        # Direção da câmera no espaço do mundo
        forward = np.array([0, 0, -1, 0])   # vetor para frente em coords locais
        cam_dir_world = GL.Mr_c @ forward
        GL.cam_direction = (cam_dir_world[0], cam_dir_world[1], cam_dir_world[2])

        # Se o headlight está ligado, a luz segue a câmera
        if GL.hasLight:
            GL.light_direction = GL.cam_direction

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Transform : ", end='')
        # if translation:
        #     print("translation = {0} ".format(translation), end='') # imprime no terminal
        # if scale:
        #     print("scale = {0} ".format(scale), end='') # imprime no terminal
        # if rotation:
        #     print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        # print("")

        GL.Mt = np.array([
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1]
        ])

        GL.Ms = np.array([
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1]
        ])

        # Ex.: rotation = [-1.0, 0.0, 0.0, 3.1415]
        qi = rotation[0] * math.sin(rotation[3]/2)
        qj = rotation[1] * math.sin(rotation[3]/2)
        qk = rotation[2] * math.sin(rotation[3]/2)
        qr = math.cos(rotation[3]/2)
        GL.Mr = np.array([
            [1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0],
            [2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr), 0],
            [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2), 0],
            [0, 0, 0, 1]
        ])
        
        GL.stack.append(GL.last_matrix)
        GL.last_matrix = GL.last_matrix @ GL.Mt @ GL.Mr @ GL.Ms
        # print(GL.last_matrix, len(GL.stack))
        # print(GL.M)
        # print(GL.last_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Saindo de Transform")

        GL.last_matrix = GL.stack.pop()
        # print(GL.last_matrix, len(GL.stack))
        # print(GL.last_matrix)


    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        
        # Pega as cores default
        r, g, b = colors["emissiveColor"]
        emissiveColor = colors["emissiveColor"]
        transparency = colors["transparency"]

        # Caracteristicas do material
        diffuseColor = colors["diffuseColor"]
        specularColor = colors["specularColor"]
        shininess = colors["shininess"]

        for i, strip in enumerate(stripCount):
            for j in range(strip-2):
                idx_0 = 3*i + j 
                idx_1 = 3*i + j + 1
                idx_2 = 3*i + j + 2
                if j%2 != 0:
                    idx_1 = 3*i + j + 2
                    idx_2 = 3*i + j + 1
                
                p0 = GL.transform_3Dto2D(point[3*idx_0], point[3*idx_0 + 1], point[3*idx_0 + 2])
                p1 = GL.transform_3Dto2D(point[3*idx_1], point[3*idx_1 + 1], point[3*idx_1 + 2])
                p2 = GL.transform_3Dto2D(point[3*idx_2], point[3*idx_2 + 1], point[3*idx_2 + 2])

                GL.draw_triangle([p0, p1, p2], r, g, b, diffuseColor=diffuseColor, specularColor=specularColor,
                    shininess=shininess, transparencia=transparency, emissiveColor=emissiveColor)

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        # gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        # print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        # Pega as cores default
        r, g, b = colors["emissiveColor"]
        emissiveColor = colors["emissiveColor"]
        transparency = colors["transparency"]

        # Caracteristicas do material
        diffuseColor = colors["diffuseColor"]
        specularColor = colors["specularColor"]
        shininess = colors["shininess"]

        # Vertices dos pontos para calculo das normais (Iluminacao)
        vertices = [np.array([point[i], point[i+1], point[i+2]], dtype=float)
            for i in range(0, len(point), 3)]

        # Calcula normais por vertice (Iluminacao)
        normais = GL.calcula_normais(vertices, index)

        count_reset = 0
        for i in range(len(index)):
            if (i + 2) == len(index):
                break

            if (index[i+2] == -1) or (count_reset > 0 and count_reset < 3):
                count_reset += 1
            else:
                idx_0 = index[i]
                idx_1 = index[i+1]
                idx_2 = index[i+2]
                if i%2 != 0:
                    idx_1 = index[i+2]
                    idx_2 = index[i+1]

                # Pontos
                p0 = GL.transform_3Dto2D(point[3*idx_0], point[3*idx_0 + 1], point[3*idx_0 + 2])
                p1 = GL.transform_3Dto2D(point[3*idx_1], point[3*idx_1 + 1], point[3*idx_1 + 2])
                p2 = GL.transform_3Dto2D(point[3*idx_2], point[3*idx_2 + 1], point[3*idx_2 + 2])

                # Normais de cada vertice (Iluminacao)
                n0 = normais[idx_0]
                n1 = normais[idx_1]
                n2 = normais[idx_2]

                GL.draw_triangle([p0, p1, p2], r, g, b,  diffuseColor=diffuseColor, specularColor=specularColor,
                    shininess=shininess, transparencia=transparency, emissiveColor=emissiveColor, vertexNormals=[n0, n1, n2])
                count_reset = 0

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        hasTexture = 0
        img_shape = 0
        image = []
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            img_shape = image.shape
            hasTexture = 1

        if colorPerVertex and not color:
            colorPerVertex = False

        # Cria uma lista com os pontos
        vertices = []
        color_vert = []     # Guarda as cores para cada vetor
        text_pontos = []    # Guarda as coord. UV (textura)
        i_text = 0
        for i in range(0, len(coord), 3):
            x = coord[i]
            y = coord[i+1]
            z = coord[i+2]
            vertices.append((x, y, z))

            if hasTexture:
                u = texCoord[i_text]
                v = texCoord[i_text+1]
                text_pontos.append((u, v))
                i_text += 2
            elif colorPerVertex:
                r = color[i]
                g = color[i+1]
                b = color[i+2]
                color_vert.append((r, g, b))

        # Pega as cores default
        r, g, b = colors["emissiveColor"]
        emissiveColor = colors["emissiveColor"]
        transparency = colors["transparency"]

        # Caracteristicas do material
        diffuseColor = colors["diffuseColor"]
        specularColor = colors["specularColor"]
        shininess = colors["shininess"]
    
        # Exemplo: 0, 1, 2, 3, 4, -1
        # Primeiro -> 0, 1, 2
        # Segundo  -> 0, 2, 3
        # Terceito -> 0, 3, 4
        reset = 1                       # Para resetar o ponto que conecta todos os outros
        conexoes = [-99, -99, -99]      # Lista para conectar os pontos
        con_color = [-99, -99, -99]     # Lista de cores para os pontos das conexoes
        con_text = [-99, -99, -99]      # Lista para conectar os pontos com textura
        count = 1
        for i in range(len(coordIndex)):
            # Apos ter encontrado um -1, reseta a lista
            if reset == 1:
                i_vertice = coordIndex[i]
                p0 = vertices[i_vertice]    # Pega ponto 0

                # Pega o Ponto P0 e transforma para "2D"
                p0 = GL.transform_3Dto2D(p0[0], p0[1], p0[2])
                conexoes = [p0, -99, -99]

                # Pega as coord UV (textura) do P0
                if hasTexture:
                    i_text = texCoordIndex[i]
                    p0_uv = text_pontos[i_text]
                    con_text = [p0_uv, -99, -99]
                # Se pontos tiverem cor
                elif colorPerVertex:
                    i_color = colorIndex[i]
                    p0_cor = color_vert[i_color]  # Pega a cor do ponto 0
                    con_color = [p0_cor, -99, -99]

                count = 1
                reset = 0
            elif coordIndex[i] == -1:
                reset = 1
            elif count == 1:
                i_vertice = coordIndex[i]

                # Pega o Ponto P1 e transforma para "2D"
                p1 = vertices[i_vertice]
                p1 = GL.transform_3Dto2D(p1[0], p1[1], p1[2])
                conexoes[1] = p1

                # Pega as coord UV (textura) do P1
                if hasTexture:
                    i_text = texCoordIndex[i]
                    p1_uv = text_pontos[i_text]
                    con_text[1] = p1_uv
                # Pega a cor do P1
                elif colorPerVertex:
                    i_color = colorIndex[i]
                    p1_cor = color_vert[i_color]
                    con_color[1] = p1_cor
                
                count += 1
            else:
                # Pega o proximo ponto válido (e sua cor)
                i_vertice = coordIndex[i]
                p2 = vertices[i_vertice]

                if hasTexture:
                    i_text = texCoordIndex[i]
                    p2_uv = text_pontos[i_text]

                    # u = p2_uv[0]
                    con_text[2] = p2_uv
                elif colorPerVertex:
                    i_color = colorIndex[i]
                    p2_cor = color_vert[i_color]
    
                    # Pega as cores do ponto
                    r = p2_cor[0]
                    g = p2_cor[1]
                    b = p2_cor[2]
                    con_color[2] = (r, g, b)

                # Pega o Ponto P2 e transforma para "2D"
                p2 = GL.transform_3Dto2D(p2[0], p2[1], p2[2])

                # Pega os pontos da lista
                p0 = conexoes[0]
                p1 = conexoes[1]

                # Faz o Triangulo
                r, g, b = diffuseColor # ANTONIO TIRA ISSO AQUI DPS, BOTEI SO PRA TESTAR
                GL.draw_triangle([p0, p1, p2], r, g, b, colorPerVertex=colorPerVertex, vertexColors=con_color, hasTexture=hasTexture, textCoords=con_text,
                    textShape=img_shape, textImg=image, diffuseColor=diffuseColor, specularColor=specularColor,
                    shininess=shininess, transparencia=transparency, emissiveColor=emissiveColor)

                # Arruma ordem para o próximo
                conexoes[1] = p2

                if hasTexture:
                    con_text[1] = p2_uv
                elif colorPerVertex:
                    con_color[1] = p2_cor
                
                count += 1

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        GL.hasLight = headlight

        if headlight:
            GL.light_intensity = 1                  # Intensidade da Headlight
            GL.light_color = (1, 1, 1)              # Color
            GL.ambientIntensity = 0.0               # ambientIntensity
            GL.light_direction = GL.cam_direction   # Direção da Headlight

            GL.headlight = True                     # Para saber que vem da headlight

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        if intensity > 0:
            GL.hasLight = True
            GL.ambientIntensity = ambientIntensity
            GL.light_color = (color[0], color[1], color[2])
            GL.light_intensity = intensity
            GL.light_direction = (direction[0], direction[1], direction[2])

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        # print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        # print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        # print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        # print("SplinePositionInterpolator : closed = {0}".format(closed))

        if set_fraction == 0:
            return keyValue[0:3]
        
        if set_fraction == 1:
            return keyValue[-3:]

        # encontrando os dois keys para usar na interpolacao:
        for i, k in enumerate(key):
            if set_fraction < k:
                i0 = i - 1
                i1 = i
                break

        t = (set_fraction - key[i0]) / (key[i1] - key[i0])
        # print(t, set_fraction, key[i0], key[i1])

        S = np.transpose(np.array([[t**3], [t**2], [t], [1]]))

        H = np.array([
            [2, -2, 1, 1],
            [-3, 3, -2, -1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ])

        p0 = [keyValue[i0 * 3], keyValue[i0 * 3 + 1], keyValue[i0 * 3 + 2]]
        p1 = [keyValue[i1 * 3], keyValue[i1 * 3 + 1], keyValue[i1 * 3 + 2]]


        if i0 - 1 < 0 and not closed:
            d0_x = 0
            d0_y = 0
            d0_z = 0
        else:   
            p0_0 = [keyValue[((i0 - 1) % len(key)) * 3], keyValue[((i0 - 1) % len(key)) * 3 + 1], keyValue[((i0 - 1) % len(key)) * 3 + 2]]
            p1_0 = [keyValue[((i0 + 1) % len(key)) * 3], keyValue[((i0 + 1) % len(key)) * 3 + 1], keyValue[((i0 + 1) % len(key)) * 3 + 2]]
            # print(i0, (i0 - 1) % len(key), (i0 + 1) % len(key))
            # print(p0_0[0], p0_0[1], p0_0[2])
            # print(p1_0[0], p1_0[1], p1_0[2])

            d0_x = (p1_0[0] - p0_0[0]) / 2
            d0_y = (p1_0[1] - p0_0[1]) / 2
            d0_z = (p1_0[2] - p0_0[2]) / 2

        if i1 + 1 == len(key) and not closed:
            d1_x = 0
            d1_y = 0
            d1_z = 0
        else:
            # print(i1, (i1 - 1) % len(key), (i1 + 1) % len(key))
            p0_1 = (keyValue[((i1 - 1) % len(key)) * 3], keyValue[((i1 - 1) % len(key)) * 3 + 1], keyValue[((i1 - 1) % len(key)) * 3 + 2])
            p1_1 = (keyValue[((i1 + 1) % len(key)) * 3], keyValue[((i1 + 1) % len(key)) * 3 + 1], keyValue[((i1 + 1) % len(key)) * 3 + 2])
            
            d1_x = (p1_1[0] - p0_1[0]) / 2
            d1_y = (p1_1[1] - p0_1[1]) / 2
            d1_z = (p1_1[2] - p0_1[2]) / 2

        p2 = [d0_x, d0_y, d0_z]
        p3 = [d1_x, d1_y, d1_z]

        C = np.array([
            p0,
            p1,
            p2,
            p3
        ])

        # print(S)
        # print(H)
        # print(C)

        value_changed = S @ H @ C

        # print(value_changed[0])
        
        return value_changed[0]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        # print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        # print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # if set_fraction == 0:
        #     return keyValue[0:4]
        
        # if set_fraction == 1:
        #     return keyValue[-4:]

        # encontrando os dois keys para usar na interpolacao:
        for i, k in enumerate(key):
            if set_fraction < k:
                i0 = i - 1
                i1 = i
                break

        p0 = [keyValue[i0 * 4], keyValue[i0 * 4 + 1], keyValue[i0 * 4 + 2], keyValue[i0 * 4 + 3]]
        p1 = [keyValue[i1 * 4], keyValue[i1 * 4 + 1], keyValue[i1 * 4 + 2], keyValue[i1 * 4 + 3]]

        # print(p0)
        # print(p1)
        orientation = [p0[i] + set_fraction * (p1[i] - p0[i])/(key[i1] - key[i0]) for i in range(len(p0))]
        # print(orientation)
        qi = orientation[0] * math.sin(orientation[3]/2)
        qj = orientation[1] * math.sin(orientation[3]/2)
        qk = orientation[2] * math.sin(orientation[3]/2)
        qr = math.cos(orientation[3]/2)
        M = np.array([
            [1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0],
            [2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr), 0],
            [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2), 0],
            [0, 0, 0, 1]
        ])
        
        return orientation

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
