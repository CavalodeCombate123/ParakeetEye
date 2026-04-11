import tkinter as tk

from operations import (
    upload_imagem,
    abrir_webcam,
    cadastrar_pessoa,
    listar_pessoas,
    exportar_banco_menu,
    importar_banco_interativo,
)

# -------------------------------
# INTERFACE
# -------------------------------
janela = tk.Tk()
janela.title("ParakeetEye")
janela.geometry("600x600")

tk.Label(janela, text="🦜 ParakeetEye", font=("Arial", 20)).pack(pady=20)
tk.Label(janela, text="Sistema de reconhecimento facial", font=("Arial", 10)).pack(pady=20)

tk.Button(janela, text="📁 Upload de Fotos", command=upload_imagem, width=30, height=3).pack(pady=15)
tk.Button(janela, text="📹 Webcam", command=abrir_webcam, width=30, height=3).pack(pady=15)
tk.Button(janela, text="👤 Cadastrar Pessoa", command=cadastrar_pessoa, width=30, height=3).pack(pady=15)
tk.Button(janela, text="📋 Ver Pessoas Cadastradas", command=lambda: listar_pessoas(janela), width=30, height=3).pack(pady=15)
tk.Button(janela, text="📦 Exportar banco de dados", command=lambda: exportar_banco_menu(janela), width=30, height=3).pack(pady=15)
tk.Button(janela, text="📥 Importar banco de dados", command=lambda: importar_banco_interativo(janela), width=30, height=3).pack(pady=15)

janela.mainloop()
