import face_recognition as fr
from engine import reconhecimento_facial, get_rostos

undefined_photo = reconhecimento_facial("base_de_dados/TonyStarkTest.jpg")

if(undefined_photo[0]):
    rosto_desconhecido = undefined_photo[1][0]
    rostos_conhecidos, nomes_dos_rostos = get_rostos()
    results = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
    print(results)

    for i in range(len(nomes_dos_rostos)):
        match = results[i]
        if(match):
            print(f"Rosto do {nomes_dos_rostos[i]} foi reconhecido.")
else:
    print("Nenhum Rosto Encontrado =(")