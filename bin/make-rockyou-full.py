# utility script that creates the full rockyou leak using the word count
# redirect this output to a file to save it. Once you do that it is recommended
# to 'sort -R' to randomize it.

with open('../data/rockyou-full.txt', 'w') as fo:

    #with open('../data/rockyou-withcount.txt', 'r', encoding="ISO-8859-1") as fi:
    with open('../data/rockyou-withcount.txt', 'rb') as fi:
        for line in fi:
            
            try:
                
                # Decode ISO-8859-1
                line = line.decode('ISO-8859-1')
                
                # Remove leading spaces
                line = line.strip() 

                # Split count and password
                count, password = line.split(' ', 1)
                
                # Remove spaces from password
                password = password.replace(" ", "")
                
                for i in range(int(count)):

                    # Output password to file
                    print(password, file=fo)
                    #fo.write(password)
                    #fo.write("¥n")
                    
            except ValueError:
                # Skipping enpty password
                continue