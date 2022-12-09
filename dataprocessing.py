import re

def cleanData(prodset):
    cleanList = {'Inch':'inch','inches':'inch','"':'inch','-inch':'inch',' inch':'inch',
                  'Hertz':'hz','hertz':'hz','Hz':'hz','HZ':'hz',' hz':'hz','-hz':'hz',
                  }
    for prod in prodset:
        # clean the title
        for word, clean in cleanList.items():
            if word in prod['title']:
                prod['title'] = prod['title'].replace(word, clean)

        # clean kvp's
        for key, feature in prod['featuresMap'].items():
            for word, clean in cleanList.items():
                if word in feature.upper():
                    prod['featuresMap'][key] = feature.replace(word, clean)
    return prodset


# Normalize the model words
def mwNormalize(set_temp):
    set_clean = set()
    normalized = {'Inch':'inch','inches':'inch','"':'inch','-inch':'inch',' inch':'inch',
                  'Hertz':'hz','hertz':'hz','Hz':'hz','HZ':'hz',' hz':'hz','-hz':'hz',
                  '-':'','\(':'','\)':'','\[':'','\]':'',
                  }
    for word in set_temp:
        for char in normalized.keys():
            word = re.sub(char, normalized[char], word)
        set_clean.update([word.upper()])
    return set_clean

# Extracting Model Words (Normalized)
def exMW_title(prod):
    reg = '([a-zA-Z0-9]*(([0-9]+[^0-9,]+)|([^0-9,]+[0-9]+))[a-zA-Z0-9]*)'
    brandlist = ['BAND & OLUFSEN', 'CONTINENTAL EDITION', 'DENVER','EDENWOOD','GRUNDIG', 'HAIER', 'HISENSE',
                 'HITACHI','HKC','HUAWEI','INSIGNIA', 'JVC','LEECO','LG','LOEWE','MEDION','MERTZ','MOTOROLA',
                  'ONEPLUS','PANASONIC','PHILIPS','RCA', 'SAMSUNG','SCEPTRE','SHARP','SKYWORTH','SONY',
                 'TCL','TELEFUNKEN','THOMSON','TOSHIBA','VESTEL','VIZIO','XIAOMI','NOKIA','ENGEL','NEVIR',
                 'TD SYSTEMS','HYUNDAI','STRONG','REALME','OPPO','METZ BLUE','ASUS','AMAZON','CECOTEC']
    set_MW_temp = set()
    title = prod['title']
    words = title.split()
    for word in words:
        temp = re.match(reg, word)
        if temp:
            set_MW_temp.update([word])
        if word.upper() in brandlist:
            set_MW_temp.update([word.upper()])
    set_MW = mwNormalize(set_MW_temp)
    return set_MW

def exMW_kpv(prod, keys):
    reg = '(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)'
    set_MW_temp = set()
    features = prod['featuresMap']
    identifiers = ['SIZE', 'WIDTH', 'HEIGHT', 'DEPTH', 'RATIO', 'WEIGHT', 'RESOLUTION', 'REFRESH']

    def bool_identifier(identifiers, key):
        for identifier in identifiers:
            if identifier in key.upper():
                return True
        return False

    for key in keys:
        if bool_identifier(identifiers,key):
            words = features[key].split()
            for word in words:
                temp = re.match(reg, word)
                if temp:
                    set_MW_temp.update([word])
    set_MW = mwNormalize(set_MW_temp)
    set_MW_final = set()
    for item in set_MW:
        final = re.sub('[^\d+\.]','', item)
        set_MW_final.update([final])
    return set_MW_final
