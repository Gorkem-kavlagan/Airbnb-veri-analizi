import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Arial'
pd.set_option('display.max_columns', 100)

print("Kütüphaneler yüklendi. Veri seti okunuyor...")

try:
    df = pd.read_csv('listings.csv.gz', compression='gzip')
    print("Veri seti başarıyla yüklendi!")
    print(f"Toplam {df.shape[0]} satır ve {df.shape[1]} sütun bulunmaktadır.")
    print("\nVeri Setinin İlk 5 Satırı:")
    print(df.head())
except FileNotFoundError:
    print("\nHATA: 'listings.csv.gz' dosyası bulunamadı.")
    print("Lütfen dosyayı indirip 'AirbnbProjesi' klasörüne taşıdığından emin ol.")

print("\n--- Veri Temizleme ve Hazırlama Aşaması Başladı ---")

columns_to_keep = [
    'neighbourhood_cleansed', 'property_type', 'room_type', 
    'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 
    'price', 'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating', 
    'latitude', 'longitude'
]
df_selected = df[columns_to_keep].copy()

print(f"\nİşlem yapılacak sütun sayısı: {len(df_selected.columns)}")

df_selected['price'] = df_selected['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df_selected.dropna(subset=['price'], inplace=True)

df_selected['bathrooms'] = df_selected['bathrooms_text'].str.extract('(\d+\.?\d*)').astype(float)

columns_to_fill = ['bedrooms', 'beds', 'bathrooms', 'review_scores_rating']
for col in columns_to_fill:
    median_value = df_selected[col].median()
    df_selected[col].fillna(median_value, inplace=True)

print("\nEksik veriler medyan değerleri ile dolduruldu.")

df_clean = df_selected.drop(columns=['bathrooms_text'])

print("\nVeri temizleme sonrası veri setinin son durumu:")
print(f"Toplam {df_clean.shape[0]} satır kaldı.")
print("\nTemizlenmiş verinin ilk 5 satırı:")
print(df_clean.head())
print("\nTemizlenmiş verinin temel istatistikleri:")
print(df_clean.describe())

print("\n--- Veri Görselleştirme Aşaması Başladı ---")

plt.figure(figsize=(12, 6))
sns.histplot(df_clean[df_clean['price'] < 10000]['price'], bins=50, kde=True)
plt.title('İstanbul Airbnb Fiyat Dağılımı (Gecelik)')
plt.xlabel('Fiyat (TL)')
plt.ylabel('Listeleme Sayısı')
plt.show()

plt.figure(figsize=(12, 8))
top_15_neighborhoods = df_clean.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=top_15_neighborhoods.index, y=top_15_neighborhoods.values, palette='viridis', hue=top_15_neighborhoods.index, legend=False)
plt.title('Semtlere Göre Ortalama Gecelik Airbnb Fiyatları (Top 15)')
plt.xlabel('Semtler')
plt.ylabel('Ortalama Fiyat (TL)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout() 
plt.show()

df_map = df_clean[df_clean['price'] < 12000].copy()
plt.figure(figsize=(12, 12))
sns.scatterplot(
    data=df_map,
    x='longitude',
    y='latitude',
    hue='price',
    palette='rocket_r',
    size='number_of_reviews',
    sizes=(10, 250),
    alpha=0.6
)
plt.title('İstanbul Haritası Üzerinde Airbnb Fiyat ve Popülerlik Dağılımı')
plt.xlabel('Boylam (Longitude)')
plt.ylabel('Enlem (Latitude)')
plt.legend(title='Fiyat (TL)')
plt.show()

print("\n--- Makine Öğrenmesi Modeli Aşaması Başladı ---")

alt_limit = df_clean['price'].quantile(0.05)
ust_limit = df_clean['price'].quantile(0.95)

print(f"Aykırı değerler filtrelenmeden önce satır sayısı: {len(df_clean)}")
df_clean = df_clean[(df_clean['price'] >= alt_limit) & (df_clean['price'] <= ust_limit)]
print(f"Fiyatı {alt_limit:.2f} TL altı ve {ust_limit:.2f} TL üstü olan evler çıkarıldı.")
print(f"Aykırı değerler filtrelendikten sonra satır sayısı: {len(df_clean)}")

print("--- Adım 4.1: Veri Hazırlama ---")

target = 'price'
X = pd.get_dummies(df_clean.drop(columns=[target]), drop_first=True)
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nVeri hazırlama tamamlandı.")
print(f"Eğitim için kullanılacak veri boyutu: {X_train.shape[0]} satır")
print(f"Test için kullanılacak veri boyutu: {X_test.shape[0]} satır")

print("\n--- Model Eğitme Aşaması Başladı ---")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model başarıyla eğitildi.")

y_pred = model.predict(X_test)

print("Test verileri için tahminler yapıldı.")

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Değerlendirme Sonuçları ---")
print(f"Ortalama Mutlak Hata (Mean Absolute Error - MAE): {mae:.2f} TL")
print(f"R-Kare (R-Squared - R²) Skoru: {r2:.2f}")

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='mako')
plt.title('Model İçin En Önemli 10 Özellik')
plt.xlabel('Önem Düzeyi')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.show()