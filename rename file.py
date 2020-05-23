import os
# Function to rename multiple files
def main():
   i = 1
   path="crop image/new/"
   c = 'pearl_millet'
   for filename in os.listdir(path):
      my_dest = c + '_single_image_' + str(i) + '.png'
      my_source =path + filename
      my_dest =path + my_dest
      # rename() function will
      # rename all the files
      os.rename(my_source, my_dest)
      i += 1

   path = "crop image/"
   for filename in os.listdir(path):
      if filename == 'new':
         my_dest = c
         my_source =path + filename
         my_dest =path + my_dest
         os.rename(my_source, my_dest)
   
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()