import os
import sys
import matplotlib.pyplot as plt
import numtoolkit as nt

"""
Симуляция аттрактора Лоренца на RK4.
dx/dt = sigma(y - x)
dy/dt = x(rho - z) - y
dz/dt = xy - beta z
"""

def main():
    print("Запуск симуляции системы Лоренца с использованием метода RK4...")
    
    # Исходные настройки
    y0 = [1.0, 1.0, 1.0]
    dt = 0.01
    steps = 5000
    
    # Расчет траектории
    trajectory = nt.integrate(nt.lorenz, y0, dt, steps, nt.rk4_step)
    
    # Распаковка осей для графика
    x = [p[0] for p in trajectory]
    y = [p[1] for p in trajectory]
    z = [p[2] for p in trajectory]
    
    # Построение 3D-визуализации
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, lw=0.7, color='royalblue', alpha=0.8)
    
    ax.set_title("Странный аттрактор Лоренца", fontsize=14)
    ax.set_xlabel("Ось X")
    ax.set_ylabel("Ось Y")
    ax.set_zlabel("Ось Z")
    
    plt.show()

if __name__ == "__main__":
    main()
