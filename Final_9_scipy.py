import tkinter as tk                                                          
from tkinter import ttk, messagebox                                            
import numpy as np                                                            
import matplotlib.pyplot as plt                                                
from matplotlib.figure import Figure                                           
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  
from scipy import integrate, signal                                            

# ==================== 1. 核心算法逻辑 ====================

def eval_force_func(t_input, expr_str):                                        # 解析输入的 F(t) 表达式
    context = {                                                                # 允许的符号白名单
        'np': np,                                                             
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,                          
        'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi,                           
        'abs': np.abs, 'power': np.power,                                      
        't': t_input                                                           
    }
    try:
        result = eval(expr_str, {"__builtins__": None}, context)                
        if isinstance(t_input, np.ndarray) and np.isscalar(result):            # 若输入为数组但返回标量
            return np.full_like(t_input, result, dtype=float)                  # 扩展为同长度数组
        return result                                                          # 返回标量或数组
    except Exception as e:
        raise ValueError(f"函数解析错误: {e}")                                   # 统一抛出解析错误

def solve_dynamics(m, k, zeta, T, N, u0, v0, force_expr):                      # 计算多种算法的时程响应
    dt = T / N                                                                 # 主时间步长
    t_array = np.linspace(0, T, N + 1)                                         # 主时间轴(含端点)

    omega0 = np.sqrt(k / m)                                                    # 自振圆频率 ω0
    c = 2 * m * omega0 * zeta                                                  # 粘滞阻尼系数 c=2mω0ζ

    def get_F(t):                                                              # 外力函数句柄
        return eval_force_func(t, force_expr)                                  

    try:
        f0 = get_F(0.0)                                                        
    except:
        f0 = 0.0                                                               # 若解析失败，回退为 0
    a0 = (f0 - c * v0 - k * u0) / m                                            # 初始加速度由平衡方程反算

    results = {}                                                               

    # --- 1. Duhamel ---
    def run_duhamel():                                                         # 杜哈梅卷积积分
        n_fine = 2000                                                          # 细分步数,用于积分精度
        t_fine = np.linspace(0, T, n_fine + 1)                                 # 细时间轴
        dt_fine = T / n_fine                                                   # 细步长
        wd = omega0 * np.sqrt(1 - zeta**2)                                     # 阻尼圆频率 ωd

        decay = np.exp(-zeta * omega0 * t_fine)                                # 指数衰减项 e^{-ζω0 t}
        u_free = decay * (u0 * np.cos(wd * t_fine) +                           # 自由响应位移(欠阻尼)
                          (v0 + zeta * omega0 * u0) / wd * np.sin(wd * t_fine))

        u_forced = np.zeros(n_fine + 1)                                        # 受迫响应容器
        pre_factor = 1.0 / (m * wd)                                            # 卷积前系数 1/(mωd)
        forces = get_F(t_fine)                                                 # 细时间轴力序列

        for i in range(1, n_fine + 1):                                         # 对每个时刻做历史积分
            taus = t_fine[:i+1]                                                # τ ∈ [0, t_i]
            p_tau = forces[:i+1]                                               # p(τ)
            t_curr = t_fine[i]                                                 # 当前 t_i

            h = np.exp(-zeta * omega0 * (t_curr - taus)) * np.sin(wd * (t_curr - taus))  # 核函数 h(t-τ)
            integrand = p_tau * h                                              # 被积函数 p(τ)h(t-τ)

            if len(taus) > 2:                                                  # 点数足够则用 Simpson
                integral_val = integrate.simpson(y=integrand, x=taus)          # 精确数值积分
            else:
                integral_val = np.trapezoid(integrand, taus)                   # 点数少则用梯形

            u_forced[i] = pre_factor * integral_val                            # 得到 u_forced(t_i)

        u_total = u_free + u_forced                                            # 总位移=自由+受迫
        v_total = np.zeros_like(u_total)                                       # 速度容器
        v_total[1:-1] = (u_total[2:] - u_total[:-2]) / (2 * dt_fine)           # 中心差分速度
        v_total[0] = v0                                                        # 边界：初速度
        v_total[-1] = (u_total[-1] - u_total[-2]) / dt_fine                    # 边界：后向差分

        a_total = (forces - c * v_total - k * u_total) / m                     # 由平衡方程反算加速度

        return (np.interp(t_array, t_fine, u_total),                           # 插值回主时间轴 u
                np.interp(t_array, t_fine, v_total),                           # 插值回主时间轴 v
                np.interp(t_array, t_fine, a_total))                           # 插值回主时间轴 a

    results['duhamel'] = run_duhamel()                                         # 运行并缓存杜哈梅结果

# --- 2. General Step Force ---
    def run_general_solution():
        
        u = np.zeros(N + 1)
        v = np.zeros(N + 1)
        a = np.zeros(N + 1)
        u[0], v[0], a[0] = u0, v0, a0
        wd = omega0 * np.sqrt(1 - zeta**2)     
        exp_zw = np.exp(-zeta * omega0 * dt)
        sin_wd = np.sin(wd * dt)
        cos_wd = np.cos(wd * dt)
        z_w = zeta * omega0
        
        for i in range(N):
            u_i = u[i]
            v_i = v[i]
            p_i = get_F(t_array[i])                                         # 假设这一步内力恒定为 p_i
            term1 = v_i + z_w * u_i                                         # 齐次解部分，由初始状态 u_i, v_i 引起的自由衰减
            u_h = exp_zw * (u_i * cos_wd + (term1 / wd) * sin_wd)
            v_h = exp_zw * (v_i * cos_wd - (u_i * omega0**2 / wd + z_w * v_i / wd) * sin_wd)             # v_h = exp * (v_i*cos - (u_i*w0^2/wd + zw*v_i/wd)*sin)
            p_k = p_i / k                                                  # 特解部分 (Particular) - 由恒定力 p_i 引起的响应
            u_p = p_k * (1.0 - exp_zw * (cos_wd + (z_w / wd) * sin_wd))    # u_p(dt) = (p/k) * [1 - exp(...)]
            v_p = (p_i / m / wd) * exp_zw * sin_wd                         # v_p = (p/k) * (wn^2/wd) * exp * sin
            u[i+1] = u_h + u_p                                             # 叠加到下一步
            v[i+1] = v_h + v_p

            p_next = get_F(t_array[i+1])
            a[i+1] = (p_next - c * v[i+1] - k * u[i+1]) / m                # m*a_{i+1} + c*v_{i+1} + k*u_{i+1} = p_{i+1}
            
            if abs(u[i+1]) > 1e15 or np.isnan(u[i+1]):                     # 发散保护
                u[i+1:] = np.nan; v[i+1:] = np.nan; a[i+1:] = np.nan
                break
                
        return u, v, a

    results['general'] = run_general_solution()                             

    # --- 3. Interpolation  ---
    def run_interpolation():                                                   # 激励插值法(闭式递推，欠阻尼)
        u = np.zeros(N + 1); v = np.zeros(N + 1); a = np.zeros(N + 1)          # 初始化结果数组
        u[0], v[0], a[0] = u0, v0, a0                                          # 初始条件

        if not (zeta < 1.0):                                                   # 非欠阻尼则此实现不适用
            u[:] = np.nan; v[:] = np.nan; a[:] = np.nan                        # 置 NaN 表示不可用
            return u, v, a                                                     

        wn = omega0                                                            # ωn
        wd = wn * np.sqrt(1 - zeta**2)                                         # ωd
        sq = np.sqrt(1 - zeta**2)                                              # √(1-ζ²)

        e  = np.exp(-zeta * wn * dt)                                           # e^{-ζωnΔt}
        s  = np.sin(wd * dt)                                                   # sin(ωdΔt)
        c0 = np.cos(wd * dt)                                                   # cos(ωdΔt)

        A = e * ( (zeta/sq) * s + c0 )                                         # 递推系数 A
        B = e * ( s / wd )                                                     # 递推系数 B

        C_force = (1.0/k) * ( (2.0*zeta)/(wn*dt) +                             # 力相关系数 C（推导自插值假设）
                   e * ( ((1.0-2.0*zeta**2)/(wd*dt) - zeta/sq)*s -
                         (1.0 + (2.0*zeta)/(wn*dt))*c0 ) )
        D_force = (1.0/k) * ( 1.0 - (2.0*zeta)/(wn*dt) +                       # 力相关系数 D
                   e * ( ((2.0*zeta**2-1.0)/(wd*dt))*s + (2.0*zeta/(wn*dt))*c0 ) )

        Ap = - e * ( (wn/sq) * s )                                             # 速度递推系数 A'
        Bp =   e * ( c0 - (zeta/sq) * s )                                      # 速度递推系数 B'

        Cp_force = (1.0/k) * (                                                # 速度-力系数 C'
            -1.0/dt
            + e * (
                ((wn/sq) + (zeta/(sq*dt))) * s
                + (1.0/dt) * c0
            )
        )

        Dp_force = (1.0/(k*dt)) * ( 1.0 - e * ( (zeta/sq)*s + c0 ) )           # 速度-力系数 D'

        for i in range(N):                                                     # 逐步递推
            p_i = get_F(t_array[i])                                            # 当前力 p_i
            p_next = get_F(t_array[i+1])                                       # 下一步力 p_{i+1}
            u[i+1] = A*u[i] + B*v[i] + C_force*p_i + D_force*p_next            # 位移递推
            v[i+1] = Ap*u[i] + Bp*v[i] + Cp_force*p_i + Dp_force*p_next        # 速度递推
            a[i+1] = (p_next - c * v[i+1] - k * u[i+1]) / m                    # 加速度反算
            if abs(u[i+1]) > 1e15 or np.isnan(u[i+1]):                         # 发散保护
                u[i+1:] = np.nan; break                                        # 置 NaN 并终止

        return u, v, a                                                         

    results['interp'] = run_interpolation()                                    # 运行并缓存插值法

    # --- 4. Central Difference ---
    def run_cd():                                                              
        u = np.zeros(N + 1); u[0] = u0                                         # 位移数组
        v = np.zeros(N + 1); v[0] = v0                                         # 速度数组
        a = np.zeros(N + 1); a[0] = a0                                         # 加速度数组

        u_prev = u0 - dt * v0 + 0.5 * a0 * (dt**2)                             # 虚拟步 u_{-1}
        factor = m/(dt**2) + c/(2*dt)                                          # 有效系数(分母)
        k_prev = m/(dt**2) - c/(2*dt)                                          # u_{i-1} 系数
        k_curr = k - 2*m/(dt**2)                                               # u_i 系数

        for i in range(N):                                                     # 显式推进 u_{i+1}
            u_next = (get_F(t_array[i]) - k_prev*u_prev - k_curr*u[i]) / factor  # 计算 u_{i+1}
            u_prev = u[i]                                                      # 滚动更新 u_{i-1} <- u_i
            u[i+1] = u_next                                                    # 写入 u_{i+1}
            if abs(u_next) > 1e15 or np.isnan(u_next):                         # 发散保护
                u[i+1:] = np.nan; break                                        # 置 NaN 并终止

        for i in range(1, N):                                                  # 回算 a,v（基于位移差分）
            a[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dt**2)                        # 加速度差分
            v[i] = (u[i+1] - u[i-1]) / (2*dt)                                  # 速度差分
        if not np.isnan(u[N]):                                                 # 末端点采用单边差分
            a[N] = (u[N] - 2*u[N-1] + u[N-2]) / (dt**2)                        # 末端加速度
            v[N] = (u[N] - u[N-1]) / dt                                        # 末端速度
        else:
            a[N], v[N] = np.nan, np.nan                                        # 若发散则标 NaN
        return u, v, a                                                         

    results['cd'] = run_cd()                                                   # 运行并缓存中心差分

    # --- 5. Newmark ---
    def run_newmark(beta, gamma):                                              # Newmark-β(隐式)
        u = np.zeros(N+1); v = np.zeros(N+1); a = np.zeros(N+1)                # 初始化数组
        u[0], v[0], a[0] = u0, v0, a0                                         

        a1 = (1 / (beta * dt**2)) * m + (gamma / (beta * dt)) * c              # 等效刚度项 a1
        a2 = (1 / (beta * dt)) * m + (gamma / beta - 1) * c                    # 等效荷载项 a2
        a3 = (1 / (2 * beta) - 1) * m + dt * (gamma / (2 * beta) - 1) * c      # 等效荷载项 a3
        K_hat = k + a1                                                         # 等效刚度 K̂

        for i in range(N):                                                     # 逐步求解
            p_next = get_F(t_array[i+1])                                       # 下一步外力
            p_hat = p_next + a1 * u[i] + a2 * v[i] + a3 * a[i]                 # 等效外力 p̂
            u_next = p_hat / K_hat                                             # 解位移(此处为标量除)

            if abs(u_next) > 1e15 or np.isnan(u_next):                         # 发散保护
                 u[i+1:] = np.nan; break                                       # 置 NaN 并终止

            u[i+1] = u_next                                                         # 写入位移
            v[i+1] = ((gamma / (beta * dt)) * (u[i+1] - u[i]) +                       # 更新速度
                     (1 - gamma / beta) * v[i] + dt * (1 - gamma / (2 * beta)) * a[i])
            a[i+1] =( (1 / (beta * dt**2)) * (u[i+1] - u[i]) -                           # 更新加速度
                     (1 / (beta * dt)) * v[i] - (1 / (2 * beta) - 1) * a[i])
        return u, v, a                                                          

    results['linear_acc'] = run_newmark(1/6, 0.5)                               # 线性加速度法(β=1/6, γ=1/2)
    results['newmark'] = run_newmark(0.25, 0.5)                                 # 平均加速度法(β=1/4, γ=1/2)

    # --- 6. Wilson-theta ---
    def run_wilson(theta=1.42):                                               
        u = np.zeros(N+1); v = np.zeros(N+1); a = np.zeros(N+1)                
        u[0], v[0], a[0] = u0, v0, a0                                          

        dt_theta = theta * dt                                                  # 扩展时间步 θΔt
        K_theta = m + (dt_theta / 2.0) * c + (dt_theta**2 / 6.0) * k           # θ 时刻等效刚度

        for i in range(N):                                                     # 主循环
            p_i = get_F(t_array[i])                                            # 当前外力
            p_next = get_F(t_array[i+1])                                       # 下一步外力
            p_theta = p_i + theta * (p_next - p_i)                             # 线性外插得到 p(t+θΔt)

            c_part = v[i] + (dt_theta / 2.0) * a[i]                            # 阻尼项预测
            k_part = u[i] + dt_theta * v[i] + (dt_theta**2 / 3.0) * a[i]       # 刚度项预测
            R_theta = p_theta - c * c_part - k * k_part                        # θ 时刻等效剩余力
            a_theta = R_theta / K_theta                                        # 求 θ 时刻加速度

            a[i+1] = a[i] + (1.0 / theta) * (a_theta - a[i])                   # 回推到 t+Δt 加速度
            v[i+1] = v[i] + (dt / 2.0) * (a[i] + a[i+1])                       # 更新速度
            u[i+1] = u[i] + dt * v[i] + (dt**2 / 6.0) * (2.0 * a[i] + a[i+1])  # 更新位移

            if abs(u[i+1]) > 1e15 or np.isnan(u[i+1]):                         # 发散保护
                 u[i+1:] = np.nan; break                                       # 置 NaN 并终止

        return u, v, a                                                         

    results['wilson'] = run_wilson()                                           
    return results, t_array, dt, omega0                                        


# ==================== 2. GUI 界面类 ====================

class DynamicsApp:
    def __init__(self, root):
        self.root = root                                                       # Tk 根窗口
        self.root.title("结构动力响应计算器")                                     
        self.root.geometry("1000x800")                                          # 窗口尺寸

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 中文字体支持
        plt.rcParams['axes.unicode_minus'] = False                              # 负号正常显示
        plt.rcParams['font.size'] = 10                                          # 全局字号

        self.last_results = None                                                # 缓存：最后一次 results
        self.last_t_arr = None                                                  # 缓存：最后一次时间轴

        self.frame_left = tk.Frame(root, width=240, bg="#f0f0f0", padx=10, pady=15)  # 左侧面板
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y)                           # 左侧固定

        # --- 1. 参数输入区 ---
        self.add_section_title("1. 结构参数")                                     # 分区标题
        self.create_input("质量 m (kg):", "44364", "entry_m")                   # m 输入框
        self.create_input("刚度 k (N/m):", "1750000", "entry_k")                # k 输入框
        self.create_input("阻尼比 ζ:", "0.05", "entry_zeta")                    # ζ 输入框

        self.add_section_title("2. 初始条件")                                     # 分区标题
        self.create_input("初始位移 u0 (m):", "0.0", "entry_u0")                # u0 输入框
        self.create_input("初始速度 v0 (m/s):", "0.0", "entry_v0")              # v0 输入框

        self.add_section_title("3. 荷载与时间")                                   # 分区标题
        self.create_input("总步数 N:", "10", "entry_N")                         # N 输入框
        self.create_input("总时长 T (s):", "1.0", "entry_T")                    # T 输入框

        tk.Label(self.frame_left, text="外荷载 F(t):", bg="#f0f0f0", anchor="w",
                 font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(5, 0))       # 外荷载标签
        self.entry_force = tk.Entry(self.frame_left, fg="blue")                 # 外荷载表达式输入
        self.entry_force.insert(0, "44482 * sin(pi*t/0.6)")                     # 默认外力
        self.entry_force.pack(fill=tk.X, pady=2)                                # 布局

        # --- 4. 显示控制区 ---
        self.add_section_title("4. 显示控制")                                     # 分区标题

        self.check_vars = {                                                     # 曲线显示开关
            'duhamel': tk.BooleanVar(value=True),                               
            'general': tk.BooleanVar(value=True),                               
            'interp': tk.BooleanVar(value=True),                               
            'cd': tk.BooleanVar(value=True),                                    
            'linear_acc': tk.BooleanVar(value=True),                            
            'newmark': tk.BooleanVar(value=True),                               
            'wilson': tk.BooleanVar(value=True)                                 
        }

        self.method_labels = {                                                  # 复选框文本
            'duhamel': '杜哈梅积分',
            'general': '一般解',                                                
            'interp': '激励插值法',
            'cd': '中心差分法',
            'linear_acc': '线性加速度',
            'newmark': 'Newmark-平均',
            'wilson': 'Wilson-θ'
        }

        chk_frame = tk.Frame(self.frame_left, bg="#f0f0f0")                   # 复选框容器
        chk_frame.pack(fill=tk.X, pady=5)                                       # 布局

        i = 0
        for key, text in self.method_labels.items():                            # 创建复选框网格
            chk = tk.Checkbutton(chk_frame, text=text, variable=self.check_vars[key],
                                 bg="#f0f0f0", anchor="w", command=self.on_toggle_visibility)  # 切换只刷新图
            chk.grid(row=i//2, column=i%2, sticky="w", padx=2, pady=2)          # 2 列布局
            i += 1

        self.btn_calc = tk.Button(self.frame_left, text="计算并绘图", command=self.on_calculate,
                                  bg="#2196F3", fg="white", font=("Arial", 12, "bold"), height=2)  # 计算按钮
        self.btn_calc.pack(pady=20, fill=tk.X)                                  # 布局

        self.lbl_status = tk.Label(self.frame_left, text="准备就绪", bg="#f0f0f0", fg="black",
                                   justify=tk.LEFT, wraplength=180)            # 状态显示
        self.lbl_status.pack(pady=5, fill=tk.X)                                 # 布局

        self.frame_right = tk.Frame(root)                                       # 右侧绘图区容器
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)         # 右侧自适应

        self.fig = Figure(figsize=(5,4), dpi=100)                               # Figure
        self.ax_u = self.fig.add_subplot(311)                                   # 位移子图
        self.ax_v = self.fig.add_subplot(312)                                   # 速度子图
        self.ax_a = self.fig.add_subplot(313)                                   # 加速度子图

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)      # 嵌入画布
        self.canvas.draw()                                                      # 初次绘制

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_right)      # 工具栏
        self.toolbar.update()                                                   # 更新工具栏
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # 布局画布

    def add_section_title(self, text):
        tk.Label(self.frame_left, text=text, bg="#e0e0e0",
                 font=("Arial", 10, "bold"), anchor="w").pack(fill=tk.X, pady=(15, 5))  # 分区标题行

    def create_input(self, label_text, default_val, var_name):
        frame = tk.Frame(self.frame_left, bg="#f0f0f0")                       # 输入行容器
        frame.pack(pady=2, fill=tk.X)                                           # 布局
        tk.Label(frame, text=label_text, bg="#f0f0f0", width=14, anchor="w").pack(side=tk.LEFT)  # 标签
        entry = tk.Entry(frame, width=10)                                       # 输入框
        entry.insert(0, default_val)                                            # 默认值
        entry.pack(side=tk.RIGHT, padx=5)                                       # 布局
        setattr(self, var_name, entry)                                          # 绑定为对象属性

    def on_toggle_visibility(self):
        if self.last_results is not None and self.last_t_arr is not None:       # 若已计算过则直接刷新
            self.plot_results(self.last_results, self.last_t_arr)               # 重画，不重复计算

    def on_calculate(self):
        try:
            m = float(self.entry_m.get())                                       
            k = float(self.entry_k.get())                                       
            zeta = float(self.entry_zeta.get())                                 
            u0 = float(self.entry_u0.get())                                     
            v0 = float(self.entry_v0.get())                                     
            N = int(self.entry_N.get())                                         
            T = float(eval_force_func(0, self.entry_T.get()))                   
            force_str = self.entry_force.get()                                  

            if N <= 0 or T <= 0 or m <= 0:                                      # 合法性检查
                messagebox.showerror("错误", "质量、步数和时长必须为正数")          # 弹窗报错
                return

            try:
                eval_force_func(0.0, force_str)                                 # 预解析外力表达式
            except Exception as e:
                messagebox.showerror("函数错误", f"荷载函数解析失败:\n{e}")          # 弹窗报错
                return

            results, t_arr, dt, omega0 = solve_dynamics(m, k, zeta, T, N, u0, v0, force_str)  # 计算主函数

            self.last_results = results                                         # 缓存结果
            self.last_t_arr = t_arr                                             # 缓存时间轴

            Tn = 2 * np.pi / omega0                                             # 自振周期 Tn
            limit_dt = Tn / np.pi                                               # CD 稳定性经验阈值

            status_text = (f"dt = {dt:.4f} s\n"
                           f"Tn = {Tn:.4f} s\n"
                           f"临界步长(CD) = {limit_dt:.4f} s")                  # 状态文本

            if dt > limit_dt:                                                   # 判断 CD 稳定性
                status_text += "\n\n[警告] 中心差分法不稳定!"
                self.lbl_status.config(text=status_text, fg="red")              # 红色警告
            else:
                status_text += "\n\n[状态] 步长安全。"
                self.lbl_status.config(text=status_text, fg="green")            # 绿色正常

            self.plot_results(results, t_arr)                                   # 绘图

        except ValueError:
            messagebox.showerror("输入错误", "请检查输入数值")                      # 输入转换失败

    def plot_results(self, results, t_arr):
        self.ax_u.clear(); self.ax_v.clear(); self.ax_a.clear()                 # 清空子图

        styles_config = {                                                       # 各方法绘图样式
            'duhamel':    ('#505050', '-',  4.0, 0, 0.3, 1),                    # 灰色打底
            'general':    ('#00CED1', '-',  2.5, 0, 0.8, 9),                    # 青色
            'interp':     ('#D62728', '-',  2.0, 4, 1.0, 10),                   # 红色
            'cd':         ('#1F77B4', '--', 1.5, 3, 0.9, 5),                    # 蓝色
            'linear_acc': ('#2CA02C', '-.', 1.5, 0, 0.9, 5),                    # 绿色
            'newmark':    ('#9467BD', ':',  2.0, 0, 0.9, 5),                    # 薰衣草
            'wilson':     ('#FF7F0E', '--', 1.5, 0, 0.9, 5)                     # 橙色
        }

        labels = {                                                              # 图例名称
            'duhamel':    '杜哈梅积分',
            'general':    '一般解 ',
            'interp':     '激励插值法',
            'cd':         '中心差分法',
            'linear_acc': '线性加速度法',
            'newmark':    'Newmark-平均',
            'wilson':     'Wilson-θ法'
        }

        for key, (u, v, a) in results.items():                                  # 遍历各方法结果
            if key not in styles_config:                                        # 无样式配置则跳过
                continue
            if key in self.check_vars and not self.check_vars[key].get():       # 未勾选则不画
                continue

            color, ls, lw, ms, alpha, zorder = styles_config[key]               # 解包样式
            lbl = labels[key]                                                   # 图例文本

            self.ax_u.plot(t_arr, u, color=color, linestyle=ls, linewidth=lw,
                           alpha=alpha, label=lbl, zorder=zorder)               # 位移曲线
            if ms > 0:
                self.ax_u.plot(t_arr, u, marker='o', color=color, markersize=ms,
                               alpha=alpha, zorder=zorder, linestyle='None')    # 位移散点

            self.ax_v.plot(t_arr, v, color=color, linestyle=ls, linewidth=lw,
                           alpha=alpha, zorder=zorder)                          # 速度曲线
            if ms > 0:
                self.ax_v.plot(t_arr, v, marker='o', color=color, markersize=ms,
                               alpha=alpha, zorder=zorder, linestyle='None')    # 速度散点

            self.ax_a.plot(t_arr, a, color=color, linestyle=ls, linewidth=lw,
                           alpha=alpha, zorder=zorder)                          # 加速度曲线
            if ms > 0:
                self.ax_a.plot(t_arr, a, marker='o', color=color, markersize=ms,
                               alpha=alpha, zorder=zorder, linestyle='None')    # 加速度散点

        self.ax_u.set_ylabel("位移 u (m)")                                      
        self.ax_u.grid(True, linestyle='--', alpha=0.4)                         
        self.ax_u.legend(loc='upper right', fontsize='small', ncol=2, framealpha=0.9)  # 图例

        self.ax_v.set_ylabel("速度 v (m/s)")                                     
        self.ax_v.grid(True, linestyle='--', alpha=0.4)                        

        self.ax_a.set_ylabel(r"加速度 a ($m/s^2$)")                               
        self.ax_a.grid(True, linestyle='--', alpha=0.4)                         
        self.ax_a.set_xlabel("时间 t (s)")                                       

        try:
            use_interp = ('interp' in results) and self.check_vars['interp'].get() and \
                         (not np.all(np.isnan(results['interp'][0])))           # 优先用插值法曲线定，激励函数插值y 范围

            if use_interp:
                u_interp = results['interp'][0]                                 # 插值法位移
                u_max = np.nanmax(u_interp); u_min = np.nanmin(u_interp)        # 最大最小
                span = u_max - u_min                                            # 幅值范围
                if span == 0:                                                   # 防止零跨度
                    span = 1.0
                padding = span * 0.15                                           # 留边
                self.ax_u.set_ylim(u_min - padding, u_max + padding)            # 设定位移 y 范围
            else:
                valid_max_u = 0.0                                               # 备用：统计可见曲线最大值
                for k_res, val in results.items():
                    if k_res in self.check_vars and self.check_vars[k_res].get():
                        curr_max = np.nanmax(np.abs(val[0]))                    # 当前方法位移最大绝对值
                        if curr_max > valid_max_u:
                            valid_max_u = curr_max
                if valid_max_u == 0:
                    valid_max_u = 1.0                                           # 若无曲线则给默认
                self.ax_u.set_ylim(-valid_max_u * 1.5, valid_max_u * 1.5)       # 对称范围
        except:
            pass                                                                # y 轴锁定异常则忽略

        self.fig.tight_layout()                                                 # 自动排版
        self.canvas.draw()                                                      # 刷新画布

if __name__ == "__main__":
    root = tk.Tk()                                                              # 创建根窗口
    app = DynamicsApp(root)                                                     # 创建应用
    root.mainloop()                                                             
