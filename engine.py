import face_recognition as fr

def reconhecimento_facial(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos

    return False, []


def get_rostos():
    rosto_conhecido = []
    nome_rosto = []

    elonMusk = reconhecimento_facial("base_de_dados/Elon.jpg")
    if(elonMusk[0]):
        rosto_conhecido.append(elonMusk[1][0])
        nome_rosto.append("Elon Musk")

    tonyStark = reconhecimento_facial("base_de_dados/Tony.jpg")
    if (tonyStark[0]):
        rosto_conhecido.append(tonyStark[1][0])
        nome_rosto.append("Tony Stark")

    return rosto_conhecido, nome_rosto

