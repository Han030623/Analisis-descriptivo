from nicegui import ui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy import stats

datos = {
    'peso': [7.2, 8.5, 9.8, 6.5, 7.5, 10.1, 11, 11, 11.1, 11.2, 11.3, 11.4, 11.4, 11.7, 12, 
             12.9, 12.9, 10.3, 9.7, 10.8, 11, 10.2, 10.5, 6.5, 6.3, 7.3, 7.5, 7.9, 8.2],
    'altura': [50, 66, 73, 72, 81, 73, 66, 75, 70, 75, 69, 76, 76, 69, 75, 64, 55, 76, 71, 
               64, 78, 70, 74, 72, 77, 51, 62, 60, 70],
    'velocidad': [10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 21.2, 22.6, np.nan, 19.9, 24.2, 
                  21, 21.4, 21.3, 22.2, np.nan, 33.8, 27.4, 25.7, 24.9, 23.1, 31.7, 36.3, 
                  38.3, 42.6, 55.4, np.nan, 58.3],
    'color': ['Blanco', 'Amarillo', 'Verde', 'Verde', 'Verde', 'Verde', 'Blanco', 'Amarillo', 
              np.nan, 'Blanco', 'Amarillo', 'Blanco', 'Verde', 'Verde', 'Amarillo', 'Amarillo', 
              'Blanco', 'Amarillo', 'Verde', 'Verde', 'Amarillo', 'Verde', 'Verde', 'Verde', 
              'Blanco', 'Blanco', np.nan, 'Amarillo', 'Verde']
}

df = pd.DataFrame(datos)

def calcular_estadisticas_descriptivas():
    """Calcula media, mediana, moda y otras estadísticas descriptivas"""
    stats_dict = {}
    for columna in ['peso', 'altura', 'velocidad']:
        datos_limpios = df[columna].dropna()
        stats_dict[columna] = {
            'media': datos_limpios.mean(),
            'mediana': datos_limpios.median(),
            'moda': stats.mode(datos_limpios, keepdims=True)[0][0],
        }
    return stats_dict

def crear_grafica_barras_absolutas():
    """Crea gráfica de barras con frecuencias absolutas"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FRECUENCIAS ABSOLUTAS', fontsize=16, fontweight='bold')
    
    peso_bins = pd.cut(df['peso'].dropna(), bins=5)
    peso_freq = peso_bins.value_counts().sort_index()
    axes[0, 0].bar(range(len(peso_freq)), peso_freq.values, color='skyblue', edgecolor='navy')
    axes[0, 0].set_xticks(range(len(peso_freq)))
    axes[0, 0].set_xticklabels([str(x) for x in peso_freq.index], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Frecuencia Absoluta')
    axes[0, 0].set_title('Peso')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    altura_bins = pd.cut(df['altura'].dropna(), bins=5)
    altura_freq = altura_bins.value_counts().sort_index()
    axes[0, 1].bar(range(len(altura_freq)), altura_freq.values, color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_xticks(range(len(altura_freq)))
    axes[0, 1].set_xticklabels([str(x) for x in altura_freq.index], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Frecuencia Absoluta')
    axes[0, 1].set_title('Altura')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    velocidad_bins = pd.cut(df['velocidad'].dropna(), bins=5)
    velocidad_freq = velocidad_bins.value_counts().sort_index()
    axes[1, 0].bar(range(len(velocidad_freq)), velocidad_freq.values, color='salmon', edgecolor='darkred')
    axes[1, 0].set_xticks(range(len(velocidad_freq)))
    axes[1, 0].set_xticklabels([str(x) for x in velocidad_freq.index], rotation=45, ha='right')
    axes[1, 0].set_ylabel('Frecuencia Absoluta')
    axes[1, 0].set_title('Velocidad')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    color_freq = df['color'].dropna().value_counts()
    axes[1, 1].bar(color_freq.index, color_freq.values, color=['yellow', 'green', 'lightgray'])
    axes[1, 1].set_ylabel('Frecuencia Absoluta')
    axes[1, 1].set_title('Color')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return f'data:image/png;base64,{img_base64}'

def crear_diagrama_pastel():
    """Crea diagrama de pastel con frecuencias relativas"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FRECUENCIAS RELATIVAS (%)', fontsize=16, fontweight='bold')
    
    peso_bins = pd.cut(df['peso'].dropna(), bins=5)
    peso_freq = peso_bins.value_counts().sort_index()
    peso_rel = (peso_freq / peso_freq.sum() * 100).round(2)
    axes[0, 0].pie(peso_rel.values, labels=[str(x) for x in peso_rel.index], autopct='%1.1f%%', 
                   startangle=90, colors=plt.cm.Pastel1.colors)
    axes[0, 0].set_title('Peso')
    
    altura_bins = pd.cut(df['altura'].dropna(), bins=5)
    altura_freq = altura_bins.value_counts().sort_index()
    altura_rel = (altura_freq / altura_freq.sum() * 100).round(2)
    axes[0, 1].pie(altura_rel.values, labels=[str(x) for x in altura_rel.index], autopct='%1.1f%%', 
                   startangle=90, colors=plt.cm.Pastel2.colors)
    axes[0, 1].set_title('Altura')
    
    velocidad_bins = pd.cut(df['velocidad'].dropna(), bins=5)
    velocidad_freq = velocidad_bins.value_counts().sort_index()
    velocidad_rel = (velocidad_freq / velocidad_freq.sum() * 100).round(2)
    axes[1, 0].pie(velocidad_rel.values, labels=[str(x) for x in velocidad_rel.index], autopct='%1.1f%%', 
                   startangle=90, colors=plt.cm.Pastel1.colors)
    axes[1, 0].set_title('Velocidad')
    
    color_freq = df['color'].dropna().value_counts()
    color_rel = (color_freq / color_freq.sum() * 100).round(2)
    axes[1, 1].pie(color_rel.values, labels=color_rel.index, autopct='%1.1f%%', 
                   startangle=90, colors=['yellow', 'green', 'lightgray'])
    axes[1, 1].set_title('Color')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return f'data:image/png;base64,{img_base64}'

def crear_poligono_frecuencias():
    """Crea polígono de frecuencias"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('POLÍGONO DE FRECUENCIAS', fontsize=16, fontweight='bold')
    
    peso_bins = pd.cut(df['peso'].dropna(), bins=5)
    peso_freq = peso_bins.value_counts().sort_index()
    peso_mid = [interval.mid for interval in peso_freq.index]
    axes[0].plot(peso_mid, peso_freq.values, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0].fill_between(peso_mid, peso_freq.values, alpha=0.3)
    axes[0].set_xlabel('Peso')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Peso')
    axes[0].grid(True, alpha=0.3)
    
    altura_bins = pd.cut(df['altura'].dropna(), bins=5)
    altura_freq = altura_bins.value_counts().sort_index()
    altura_mid = [interval.mid for interval in altura_freq.index]
    axes[1].plot(altura_mid, altura_freq.values, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].fill_between(altura_mid, altura_freq.values, alpha=0.3)
    axes[1].set_xlabel('Altura')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Altura')
    axes[1].grid(True, alpha=0.3)
    
    velocidad_bins = pd.cut(df['velocidad'].dropna(), bins=5)
    velocidad_freq = velocidad_bins.value_counts().sort_index()
    velocidad_mid = [interval.mid for interval in velocidad_freq.index]
    axes[2].plot(velocidad_mid, velocidad_freq.values, 'o-', linewidth=2, markersize=8, color='red')
    axes[2].fill_between(velocidad_mid, velocidad_freq.values, alpha=0.3)
    axes[2].set_xlabel('Velocidad')
    axes[2].set_ylabel('Frecuencia')
    axes[2].set_title('Velocidad')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return f'data:image/png;base64,{img_base64}'

def calcular_frecuencias_acumuladas():
    """Calcula tablas de frecuencias acumuladas"""
    acumuladas = {}
    for columna in ['peso', 'altura', 'velocidad']:
        datos_limpios = df[columna].dropna()
        bins = pd.cut(datos_limpios, bins=5)
        freq = bins.value_counts().sort_index()
        freq_acum = freq.cumsum()
        freq_rel_acum = (freq_acum / freq_acum.iloc[-1] * 100).round(2)
        acumuladas[columna] = pd.DataFrame({
            'Intervalo': [str(x) for x in freq.index],
            'Frecuencia_Absoluta': freq.values,
            'Frecuencia_Acumulada': freq_acum.values,
            'Frecuencia_Relativa_Acumulada_%': freq_rel_acum.values
        })
    return acumuladas

@ui.page('/')
def main_page():
    """Página principal de la aplicación"""
    
    with ui.header().classes('bg-primary text-white').props('dense'):
        ui.label('ANÁLISIS ESTADÍSTICO COMPLETO').classes('text-h6')
    
    with ui.column().classes('q-pa-md'):
        ui.label('Sistema de Análisis de Datos').classes('text-h4 q-mb-md')
        ui.label('Peso, Altura, Velocidad y Color').classes('text-subtitle1 q-mb-lg text-grey-7')
        
        with ui.tabs().classes('w-full') as tabs:
            tab_datos = ui.tab('Datos')
            tab_estadisticas = ui.tab('Estadísticas')
            tab_barras = ui.tab('Frec. Absolutas')
            tab_pastel = ui.tab('Frec. Relativas')
            tab_poligono = ui.tab('Polígono')
            tab_acumuladas = ui.tab('Acumuladas')
        
        with ui.tab_panels(tabs, value=tab_datos).classes('w-full'):
            
            with ui.tab_panel('Datos'):
                ui.label('Tabla de Datos Original').classes('text-h5 q-mb-md')
                ui.table(
                    columns=[{'name': col, 'label': col.capitalize(), 'field': col, 'align': 'center'} 
                             for col in df.columns],
                    rows=df.to_dict('records'),
                    pagination=10
                ).classes('w-full')
                ui.label(f'Total de registros: {len(df)}').classes('q-mt-md text-grey-7')
                ui.label(f'Valores faltantes en velocidad: {df["velocidad"].isna().sum()}').classes('text-grey-7')
                ui.label(f'Valores faltantes en color: {df["color"].isna().sum()}').classes('text-grey-7')
            
            with ui.tab_panel('Estadísticas'):
                ui.label('Medidas de Tendencia Central').classes('text-h5 q-mb-md')
                stats = calcular_estadisticas_descriptivas()
                for variable, valores in stats.items():
                    with ui.card().classes('w-full q-mb-md'):
                        ui.label(f'{variable.upper()}').classes('text-h6 text-primary')
                        ui.separator()
                        ui.label(f'Media: {valores["media"]:.4f}').classes('text-body1')
                        ui.label(f'Mediana: {valores["mediana"]:.4f}').classes('text-body1')
                        ui.label(f'Moda: {valores["moda"]:.4f}').classes('text-body1')

            
            with ui.tab_panel('Frec. Absolutas'):
                ui.label('Gráfica de Barras - Frecuencias Absolutas').classes('text-h5 q-mb-md')
                ui.image(crear_grafica_barras_absolutas()).classes('w-full')
            
            with ui.tab_panel('Frec. Relativas'):
                ui.label('Diagrama de Pastel - Frecuencias Relativas').classes('text-h5 q-mb-md')
                ui.image(crear_diagrama_pastel()).classes('w-full')
            
            with ui.tab_panel('Polígono'):
                ui.label('Polígono de Frecuencias').classes('text-h5 q-mb-md')
                ui.image(crear_poligono_frecuencias()).classes('w-full')
            
            with ui.tab_panel('Acumuladas'):
                ui.label('Tablas de Frecuencias Acumuladas').classes('text-h5 q-mb-md')
                acumuladas = calcular_frecuencias_acumuladas()
                with ui.tabs().classes('w-full') as tabs_acum:
                    tab_peso_acum = ui.tab('Peso')
                    tab_altura_acum = ui.tab('Altura')
                    tab_velocidad_acum = ui.tab('Velocidad')
                with ui.tab_panels(tabs_acum, value=tab_peso_acum).classes('w-full'):
                    with ui.tab_panel('Peso'):
                        ui.table(
                            columns=[{'name': col, 'label': col.replace('_', ' ').title(), 
                                     'field': col, 'align': 'center'} for col in acumuladas['peso'].columns],
                            rows=acumuladas['peso'].to_dict('records')
                        ).classes('w-full')
                    with ui.tab_panel('Altura'):
                        ui.table(
                            columns=[{'name': col, 'label': col.replace('_', ' ').title(), 
                                     'field': col, 'align': 'center'} for col in acumuladas['altura'].columns],
                            rows=acumuladas['altura'].to_dict('records')
                        ).classes('w-full')
                    with ui.tab_panel('Velocidad'):
                        ui.table(
                            columns=[{'name': col, 'label': col.replace('_', ' ').title(), 
                                     'field': col, 'align': 'center'} for col in acumuladas['velocidad'].columns],
                            rows=acumuladas['velocidad'].to_dict('records')
                        ).classes('w-full')
    
    with ui.footer().classes('bg-grey-3'):
        ui.label('Desarrollado con NiceGUI - Python | 2026').classes('text-grey-7')

import os

if __name__ in {"__main__", "__mp_main__"}:
    # Render asigna el puerto automáticamente en la variable PORT
    port = int(os.environ.get('PORT', 8080))
    ui.run(title='Análisis Estadístico', host='0.0.0.0', port=port, reload=False)
