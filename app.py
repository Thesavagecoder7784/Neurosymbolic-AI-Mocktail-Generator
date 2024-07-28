# Import required libraries
from flask import Flask, request, render_template
import numpy as np # linear algebra
import tensorflow as tf
import pandas as pd
import itertools
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

tf.keras.backend.set_image_data_format('channels_last')

# Load the dataset
df = pd.read_csv('Mocktail data v6.csv')

# Define possible flavor profiles
flavors = ['Sweet', 'Sour', 'Bitter', 'Refreshing', 'Creamy']

# Set complimentary flavours
complementary_flavors={
    'Sweet':'Sour',
    'Sour':'Sweet',
    'Bitter':'Refreshing',
    'Refreshing':'Bitter',
    'Creamy':'Refreshing'
}

@app.route('/')
def index():
    return render_template('index.html')

# Display options to the user
print("Hey there! We've got some amazing options for you to choose from. Take a look at these options:")
print("1. Sweet & Sour - for those who like it classic")
print("2. Creamy & Refreshing - because life's too short to be anything else")
print("3. Bitter & Refreshing - for the adventurous souls out there")
print("4. Custom - because you're a rebel, and we like that!")

# Get user's choice
choice = input("\nAlright, let's get to it. What's your poison? Enter the number of your preferred option:")

# Process user's choice
if choice == "1":
    flavour_1 = "Sweet"
    flavour_2 = "Sour"
    print("Ah, the classic sweet and sour combo. You have a refined palate!")
    preferred_flavors = [flavour_1, flavour_2]
elif choice == "2":
    flavour_1 = "Creamy"
    flavour_2 = "Refreshing"
    print("Creamy and refreshing? You're definitely a person who knows how to keep it cool!")
    preferred_flavors = [flavour_1, flavour_2]
elif choice == "3":
    flavour_1 = "Bitter"
    flavour_2 = "Refreshing"
    print("Bitter and refreshing? You like to live dangerously, don't you?")
    preferred_flavors = [flavour_1, flavour_2]
elif choice == "4":
    flavour_1 = input('What is the first flavor profile would you like? (Sweet, Sour, Bitter, Refreshing, Creamy)')
    if flavour_1 not in flavors:
        print("That is not a valid flavour profile (atleast according to my program")
        flavour_1 = input("Please enter a valid flavour profile - Sweet, Sour, Bitter, Refreshing, Creamy")

    flavour_2 = input('What is the second flavor profile would you like? (Sweet, Sour, Bitter, Refreshing, Creamy) (Leave blank if you only want one flavor profile)')
    if flavour_2:
        if flavour_2 not in flavors:
            print("That is a not valid flavour profile")
            flavour_2 = input("Please enter a valid flavour profile")


        if flavour_1 == flavour_2:
            print("Both flavors can't be the same. Please try again.")
            flavour_2 = input('Press Enter if you only want one')

        # Recommend complementary flavor profile if only one flavor is sweet or sour
        if flavour_1 in ['Sweet', 'Sour'] and flavour_2 not in ['Sweet', 'Sour']:
            print(f"You chose {flavour_1} as your first flavor profile. We recommend {complementary_flavors[flavour_1]} as your second flavor profile.")
            ans = input("Type 'Yes' if you want to this new combination - Sweet & Sour or if you type anything else, we will go ahead with your choice")
            if(ans.lower() == 'yes'):
                flavour_2 = complementary_flavors[flavour_1]
            else:
                flavour_2 = flavour_2
        elif flavour_2 in ['Sweet', 'Sour'] and flavour_1 not in ['Sweet', 'Sour']:
            print(f"You chose {flavour_2} as your second flavor profile. We recommend {complementary_flavors[flavour_2]} as your first flavor profile.")
            ans = input("Type 'Yes' if you want to this new combination - Sweet & Sour or if you type anything else, we will go ahead with your choice")
            if(ans.lower()  == 'yes'):
                flavour_1 = complementary_flavors[flavour_2]
            else:
                flavour_1 = flavour_1
        elif flavour_1 in ['Bitter', 'Refreshing'] and flavour_2 not in ['Bitter', 'Refreshing']:
            print(f"You chose {flavour_1} as your first flavor profile. We recommend {complementary_flavors[flavour_1]} as your second flavor profile.")
            ans = input("Type 'Yes' if you want to this new combination - Bitter & Refreshing or if you type anything else, we will go ahead with your choice")
            if(ans.lower()  == 'yes'):
                flavour_2 = complementary_flavors[flavour_2]
            else:
                flavour_2 = flavour_2
        elif flavour_2 in ['Bitter', 'Refreshing'] and flavour_1 not in ['Bitter', 'Refreshing']:
            print(f"You chose {flavour_2} as your second flavor profile. We recommend {complementary_flavors[flavour_2]} as your first flavor profile.")
            ans = input("Type 'Yes' if you want to this new combination - Bitter & Resfreshing or if you type anything else, we will go ahead with your choice")
            if(ans.lower()  == 'yes'):
                flavour_1 = complementary_flavors[flavour_2]
            else:
                flavour_1 = flavour_1
        elif flavour_2 in ['Creamy', 'Refreshing'] and flavour_1 not in ['Creamy', 'Refreshing']:
            print(f"You chose {flavour_2} as your second flavor profile. We recommend {complementary_flavors[flavour_2]} as your first flavor profile.")
            ans = input("Type 'Yes' if you want to this new combination - Creamy & Refreshing or if you type anything else, we will go ahead with your choice")
            if(ans.lower()  == 'yes'):
                flavour_1 = complementary_flavors[flavour_2]
            else:
                flavour_1 = flavour_1

        # Print preferred flavor profiles
        print(f"Your preferred flavor profiles are: {flavour_1} and {flavour_2}")
        print("Custom, eh? You're the adventurous type, I like it!")

        preferred_flavors = [flavour_1, flavour_2]
    else:
        print(f"You have chosen only {flavour_1} flavor profile.")
        preferred_flavors = [flavour_1]
    
# Check if the ingredient is part of the users preferred flavour profile(s)
def has_flavor(ingredient, flavor_profiles, df):
    # Check if the ingredient is in the dataset
    if not df['Ingredient 1'].isin([ingredient]).any():
        return False
    
    # Get the number of ingredients in the row
    num_ingredients = df.loc[df['Ingredient 1'] == ingredient].shape[0]
    
    # Check if the ingredient has the desired flavor profile in any of its columns
    for i in range(num_ingredients):
        row = df.loc[df['Ingredient 1'] == ingredient].iloc[i]
        if any(flavor in [row['Flavor Profile 1'], row['Flavor Profile 2']] for flavor in flavor_profiles):
            return True
    
    # If none of the ingredients have the desired flavor profile, return False
    return False
    
# Return all ingredients with users preferred flavours
def filter_by_flavor(df, preferred_flavors):
    matching_ingredients = set()
    for flavor in preferred_flavors:
        # Filter the dataset to find ingredients with the desired flavor profile
        filtered_df = df[(df.apply(lambda x: has_flavor(x['Ingredient 1'], [flavor], df) or 
                                   has_flavor(x['Ingredient 2'], [flavor], df) or 
                                   has_flavor(x['Ingredient 3'], [flavor], df) if len(x) > 0 else False, axis=1))]
        matching_ingredients |= set(filtered_df['Ingredient 1'].unique()) | set(filtered_df['Ingredient 2'].unique()) | set(filtered_df['Ingredient 3'].unique())
    return matching_ingredients

# Filter the dataframe by preferred flavors and getting all ingredients for either flavour profile

no_ing = input("Any ingredients you dont want")
matching_ingredients = filter_by_flavor(df, preferred_flavors)
matching_ingredients = list(matching_ingredients)[:60]
for ingredient in matching_ingredients:
    if ingredient != no_ing:
        ingredient = ingredient.capitalize()
print("Matching ingredients", matching_ingredients)
print()
print("Total number of matching ingredients", len(matching_ingredients))

# Generate all possible combinations of 3 ingredients from the set
possible_combinations = []

for i in range(len(matching_ingredients)):
    for j in range(i+1, len(matching_ingredients)):
        for k in range(j+1, len(matching_ingredients)):
            combination = [matching_ingredients[i], matching_ingredients[j], matching_ingredients[k]]
            if combination not in possible_combinations:
                possible_combinations.append(combination)
new_list = []
for combination in possible_combinations:
    sorted_combination = tuple(sorted(combination))
    if sorted_combination not in new_list:
        new_list.append(sorted_combination)
        
print()
print("All possible combinations")
print("Total number of all possible combinations", len(new_list))
print()

#This is a function that checks if a given combination of ingredients is feasible, based on a set of rules.
#The function takes two inputs: the combination of ingredients, and the flavor profile of the drink.
#The function checks if any of the ingredients in the combination are incompatible with each other or with the given flavor profile.
#If any of the rules are violated, the function returns False.
#Otherwise, the function returns True.

flavor_1 = preferred_flavors[0]
flavor_2 = preferred_flavors[1]
def is_feasible(combination, flavor_profile):
    # Based on flavour profiles
    if(len(flavor_profile)==1):
        flavor_1 = flavor_profile[0]
        flavor_2 = ""
    else:
        flavor_1 = flavor_profile[0]
        flavor_2 = flavor_profile[1]
        
    sweetc = 0
    creamc = 0   
    bitterc = 0
    refreshingc = 0
    sourc = 0
    refreshingcc = 0
    
    if(len(flavor_profile)==1):
        if(flavor_1 == "Creamy"):
            if 'Jalapeno' in combination or 'Soda Water' in combination:
                return False
            cream = ['Yogurt', 'Coconut Milk', 'Milk', 'Coconut Cream']
            for ingredient in combination:
                if ingredient in cream:
                    creamc = creamc + 1
            if(creamc == 0):
                    return False
                
        elif(flavor_1=="Sour"):
            if 'Coconut Cream' in combination:
                return False
            notsour = ['Yogurt', 'Cinnamon Stick', 'Orange Peel', 'Apple Cider', 'Maraschino Cherry', 'Star Anise', 'Worcestershire Sauce', 'Carrot Juice', 'Beetroot Juice', 'Passionfruit Juice', 'Spinach', 'Cola', 'Raspberries', 'Vanilla Extract', 'Cherry Syrup', 'Tabasco Sauce', 'Celery Juice', 'Honey', 'Grenadine', 'Mango']
            for ingredient in combination:
                    if ingredient in notsour:
                        return False
        
        elif(flavor_1=='Sweet'):         
            if 'Sweet' in flavor_profile:
                if 'Jalapeno' in combination or 'Grapefruit Juice' in combination or 'Lime Juice' in combination:
                    return False
    
    #Ensuring we get either one or two sweet ingredients in a mocktail
    if(flavor_1 == "Sweet" or flavor_2 == "Sweet"):
        sweet = ['Peach Nectar', 'Mango juice', 'Strawberry', 'Peach Tea', 'Honeydew Melon Juice', 'Orange juice', 'Honey', 'Kiwi', 'Peach slices', 'Peach Juice', 'Apple Cider', 'Carrot Juice', 'Peach Syrup', 'Milk', 'Vanilla Extract', 'Simple Syrup, Lime', 'Agave syrup', 'Agave Syrup', 'Kiwi Juice', 'Coconut Cream', 'Passionfruit syrup']
        for ingredient in combination:
            if ingredient in sweet:
                sweetc = sweetc + 1
        if(sweetc<1 or sweetc>2):
            return False
    
    #Ensuring we get either one or two sour ingredients in a mocktail
    if(flavor_1 == "Sour" or flavor_2 == "Sour"):
        sour = ['Sage Leaves', 'Lemon Juice', 'Honeydew Melon Juice', 'Mint leaves', 'Ginger syrup', 'Pomegranate juice', 'Lemonade', 'Blueberries', 'Honey', 'Strawberries', 'Pineapple', 'Blueberry syrup', 'Watermelon', 'Simple syrup', 'Jalapeno Syrup', 'Mint syrup', 'Basil syrup', 'Lime Juice', 'Lime juice', 'Sage leaves', 'Watermelon Juice', 'Cherry Syrup', 'Lemon juice', 'Ice', 'Cilantro syrup', 'Lavender syrup', 'Grapefruit Slice', 'Cherry syrup', 'Blood orange juice', 'Peach Slice', 'Mint Leaves', 'Cherry', 'Vanilla extract', 'Cherry syrup', 'Dandelion Root Tea', 'Pineapple Slice', 'Grenadine', 'Blackberries', 'Simple Syrup']
        for ingredient in combination:
            if ingredient in sour:
                sourc = sourc + 1
        if(sourc<1 or sourc>2):
            return False
    
    #Ensuring we get either one or two creamy ingredients in a mocktail
    if(flavor_1 == 'Creamy' or flavor_2 == 'Creamy'):
        if 'Jalapeno' in combination or 'Soda Water' in combination:
            return False
        cream = ['Yogurt', 'Coconut Milk', 'Milk', 'Coconut Cream']
        for ingredient in combination:
            if ingredient in cream:
                creamc = creamc + 1
        if(creamc<1 or creamc>2):
            return False
    
    #Ensuring we get either one or two bitter ingredients in a mocktail
    if(flavor_1 == 'Bitter' or flavor_2 == 'Bitter'):
        bitter = ['Celery Juice', 'Beetroot Juice', 'Tonic Water', 'Lemon Juice', 'Grapefruit Juice', 'Pomegranate juice', 'Club Soda', 'Cucumber Juice', 'Jalapeno Syrup', 'Carrot Juice', 'Lime Juice', 'Spinach', 'Lemon juice', 'Turmeric Juice']
        for ingredient in combination:
            if ingredient in bitter:
                bitterc = bitterc + 1
        if(bitterc<1 or bitterc>2):
            return False

    #Ensuring we get either one or two refreshing ingredients in a mocktail    
    if(flavor_1 == 'Refreshing' or flavor_2 == 'Refreshing'):
        refreshing = ['Celery Juice', 'Peach Nectar', 'Sage Leaves', 'Ginger Ale', 'Lemon', 'Tonic Water', 'Mango juice', 'Strawberry', 'Peach Tea', 'Lemon Juice', 'Honeydew Melon Juice', 'Orange juice', 'Lime', 'Mint leaves', 'Ginger syrup', 'Cucumber Juice', 'Grapefruit Juice', 'Pomegranate juice', 'Lemonade', 'Blueberries', 'Club Soda', 'Peach slices', 'Honey', 'Kiwi', 'Coconut Water', 'Ginger ale', 'Strawberries', 'Pineapple Juice', 'Watermelon', 'Basil Leaves', 'Club soda', 'Iced tea', 'Peach Juice', 'Soda Water', 'Lime Juice', 'Lime juice', 'Mint', 'Thyme syrup', 'Watermelon Juice', 'Spinach', 'Milk', 'Lemon juice', 'Vanilla Extract', 'Ice', 'Simple Syrup, Lime', 'Agave syrup', 'Sparkling Water', 'Agave Syrup', 'Kiwi Juice', 'Ginger Juice', 'Coconut Cream', 'Passionfruit syrup', 'Turmeric Juice', 'Grapefruit Slice']
        for ingredient in combination:
            if ingredient in refreshing:
                refreshingc = refreshingc + 1
        if(refreshingc<1 or refreshingc>2):
            return False
        
    if(len(flavor_profile)==2):
        #Sweet and Sour
        if(flavor_1 in ["Sweet","Sour"] and flavor_2 in ["Sweet", "Sour"]):
            if 'Milk' in combination or 'Coconut Cream' in combination:
                return False
            if (sweetc!=1 and sourc!=1):
                return False
            ssb = 0
            balancing_ingredients = ['Blood orange juice', 'Grapefruit juice', 'Lemon Juice', 'Lemonade', 'Peach Juice', 'Passionfruit juice', 'Pineapple Juice', 'Orange juice', 'Lime', 'Pomegranate juice']
            for ingredient in combination:
                if ingredient in balancing_ingredients:
                    ssb = ssb + 1
            if(ssb<1 or ssb>1):
                return False
            
        #Creamy and Refreshing
        if(flavor_1 in ["Creamy", "Refreshing"] and flavor_2 in ["Creamy", "Refreshing"]):
            if 'Grenadine' in combination and 'Milk' in combination:
                return False
            cream = ['Yogurt', 'Coconut Milk', 'Coconut Cream']
            for ingredient in combination:
                if ingredient in cream:
                    creamc = creamc + 1
            if(creamc == 0):
                return False
            
            refreshingcc = 0
            refreshing = ['Rooibos tea', 'Grapefruit juice', 'Honeydew Melon Juice', 'Peach Tea', 'Ginger syrup', 'Turmeric Juice', 'Ginger Beer', 'Lemonade', 'Mint leaves', 'Lavender', 'Passionfruit Juice', 'Mango', 'Tonic Water', 'Orange Juice', 'Lemon', 'Jalapeno Syrup', 'Coconut Water', 'Blackberries', 'Coconut Milk', 'Spinach', 'Mint', 'Grapefruit Slice', 'Grenadine', 'Honey', 'Soda water', 'Simple Syrup, Lime', 'Chai Tea', 'Coconut Cream', 'Mango juice', 'Agave Syrup', 'Pomegranate juice', 'Coconut water', 'Peach Nectar', 'Passionfruit syrup', 'Kiwi', 'Cream Cheese', 'Mint Leaves', 'Lavender syrup', 'Blueberries', 'Club soda', 'Celery Juice', 'Peach Juice', 'Peach Syrup', 'Lime', 'Orange juice', 'Watermelon', 'Sparkling Water', 'Pineapple Juice', 'Raspberry Syrup', 'Thyme syrup', 'Peach Slice', 'Iced tea', 'Yogurt']
            for ingredient in combination:
                if ingredient in refreshing:
                    refreshingcc = refreshingcc + 1
            if(refreshingcc<1 or refreshingcc>2):
                return False
            
            crb = 0
            balancing = ['Club Soda', 'Rooibos tea', 'Ginger syrup', 'Turmeric Juice', 'Ginger Beer', 'Ice', 'Simple Syrup', 'Mint leaves', 'Lavender', 'Tonic Water', 'Lemon', 'Jalapeno Syrup', 'Coconut Water', 'Spinach', 'Mint', 'Grenadine', 'Honey', 'Soda water', 'Simple Syrup, Lime', 'Chai Tea', 'Agave Syrup', 'Passionfruit syrup', 'Mint Leaves', 'Lavender syrup', 'Club soda', 'Celery Juice', 'Vanilla extract', 'Lime', 'Sparkling Water', 'Vanilla Extract', 'Raspberry Syrup', 'Thyme syrup', 'Iced tea', 'Yogurt']
            for ingredient in combination:
                if ingredient in balancing:
                    crb = crb + 1
            if(crb<1 or crb>1):
                return False
        
        #Bitter and Refreshing
        if(flavor_1 in ["Bitter", "Refreshing"] and flavor_2 in ["Bitter", "Refreshing"]):
            bitter = ['Tonic Water', 'Grapefruit Slice', 'Club soda', 'Lime', 'Sparkling Water', 'Pineapple Juice']
            refreshing = ['Rooibos tea', 'Grapefruit juice', 'Honeydew Melon Juice', 'Peach Tea', 'Turmeric Juice', 'Lemonade', 'Mint leaves', 'Lavender', 'Passionfruit Juice', 'Mango', 'Orange Juice', 'Lemon', 'Blackberries', 'Spinach', 'Mint', 'Grapefruit Slice', 'Honey', 'Soda water', 'Agave Syrup', 'Pomegranate juice', 'Coconut water', 'Peach Nectar', 'Passionfruit syrup', 'Kiwi', 'Mint Leaves', 'Lavender syrup', 'Blueberries', 'Lime', 'Watermelon', 'Thyme syrup', 'Iced tea']

            bitterc = 0
            for ingredient in combination:
                if ingredient in bitter:
                    bitterc += 1
            if bitterc != 1:
                return False

            refreshingcc = 0
            for ingredient in combination:
                if ingredient in refreshing:
                    refreshingcc += 1
            if refreshingcc < 1 or refreshingcc > 2:
                return False

            brb = 0
            balancing = ['Ginger syrup', 'Ginger Beer', 'Ice', 'Simple Syrup', 'Jalapeno Syrup', 'Coconut Water', 'Mint', 'Grenadine', 'Soda water', 'Simple Syrup, Lime', 'Chai Tea', 'Passionfruit syrup', 'Mint Leaves', 'Club soda', 'Vanilla extract', 'Sparkling Water', 'Vanilla Extract', 'Raspberry Syrup', 'Peach Slice', 'Yogurt']
            for ingredient in combination:
                if ingredient in balancing:
                    brb += 1
            if brb != 1:
                return False

                
    # Removing all combinations with ingredients which do not work
    # Avoid using both banana and blueberries together, as their flavors may clash/they may produce an unappetizing texture.
    if 'Banana' in combination and ('Blueberries' in combination or 'Raspberries' in combination):
        return False  
    
    # Avoid combining maraschino cherry with lime juice, as they may not complement each other well.
    if 'Maraschino Cherry' in combination and ('Lime Juice' in combination or 'Cola' in combination):
        return False  
    
    # Do not use milk with any citrus-based ingredients, such as lime juice, as the acidity may cause the milk to curdle.
    if 'Milk' in combination and ('Lime Juice' in combination or 'Pineapple Juice' in combination):
        return False  
    
    # Avoid using both coconut cream and cola, as the sweetness and creaminess of coconut may not work well with the carbonation of cola.
    if 'Coconut Cream' in combination and ('Cola' in combination or 'Simple Syrup' in combination):
        return False 
    
    # Avoid using both pineapple juice or blueberries and lime juice, as their flavors may clash.
    if ('Pineapple Juice' in combination or 'Blueberries' in combination) and 'Lime Juice' in combination:
        return False  
    
    # Do not use both blueberries and maraschino cherry or coconut cream together, as their flavors may clash.
    if 'Blueberries' in combination and ('Maraschino Cherry' in combination or 'Coconut Cream' in combination):
        return False  
    
    # Avoid using both simple syrup and sugar with coconut cream, as the sweetness may become overpowering.
    if 'Coconut cream' in combination and ('Simple Syrup' in combination or 'Sugar' in combination):
        return False 
    
    # Avoid using both maraschino cherry/simple syrup/blueberry  and sugar together, as the sweetness may become overpowering.
    if ('Blueberries' in combination or 'Maraschino Cherry' in combination or'Simple Syrup' in combination)and 'Sugar' in combination:
        return False  

    # Avoid using both cola and lime juice together, as their flavors may clash.
    if ('Cola' in combination or 'Lemon' in combination) and 'Lime Juice' in combination:
        return False
    
    # Remove any combination that contains both pineapple and cranberry juice as they have strong, distinct flavors that may clash and be overpowering when combined.
    if 'Pineapple Juice' in combination and 'Cranberry Juice' in combination:
        return False
    
    #Remove any combination that contains both sprite and sugar as sprite is already sweetened and adding more sugar may make the drink too sweet.
    if 'Sprite' in combination and ('Club Soda' in combination or 'Sugar' in combination):
        return False
    
    # Remove any combination that includes both kiwi juice and grapefruit slice, as the flavors may not complement each other.
    if ('Kiwi Juice' in combination or 'Banana' in combination or 'Mango' in combination) and 'Grapefruit Slice' in combination:
        return False
    
    # Remove any combination that includes both mango and mint leaves/blueberries, as the flavors may clash.
    if 'Mango' in combination and ('Mint Leaves' in combination or  'Blackberries' in combination):
        return False
    
    # Remove any combination that includes both apple cider and coconut milk, as the flavors may clash.
    if ('Apple Cider' in combination or 'Raspberry' in combination or 'Cinnamon Stick' in combination) and 'Coconut Milk' in combination:
        return False
    
    # Remove combinations that include both citrus and dairy, as they can curdle.
    if ('Grapefruit' in combination or 'Lemon' in combination or 'Lime' in combination or 'Orange' in combination or 'Tangerine' in combination) and ('Milk' in combination or 'Yogurt' in combination):
        return False
    
    # Remove combinations that include both mint and cinnamon stick, as they can clash in flavor/they may produce a strange aftertaste.
    if 'Mint Leaves' in combination and ('Cinnamon Stick' in combination or 'Star Anise' in combination):
        return False
    
    # Remove combinations that include both kiwi juice and apple cider, as they may produce a gritty texture.
    if 'Kiwi Juice' in combination and 'Apple Cider' in combination:
        return False
    
    # No combination should contain both Sage Leaves and Maraschino Cherry.
    if 'Sage Leaves' in combination and ('Maraschino Cherry' in combination or 'Tonic Water' in combination):
        return False
    
    # Avoid combining Lime Juice and Simple Syrup, Lime.
    if 'Lime juice' in combination and 'Simple Syrup, Lime' in combination:
        return False
    
    # No combination should contain both Jalapeno and Raspberries.
    if 'Jalapeno' in combination and 'Raspberries' in combination:
        return False
    
    # Avoid combining Maraschino Cherry or Grenadine with Lime Juice or Mint Leaves.
    if ('Maraschino Cherry' in combination or 'Grenadine' in combination) and ('Lime Juice' in combination or 'Mint Leaves' in combination):
        return False
    
    # Avoid combining Pineapple Juice or Coconut Cream with Soda Water or Lemonade.
    if ('Pineapple Juice' in combination or 'Coconut Cream' in combination) and ('Soda Water' in combination or 'Lemonade' in combination):
        return False
    
    # Avoid combining Tomato Juice with Pineapple Juice, Coconut Cream, Lemonade, Worcestershire Sauce, Tabasco Sauce, Cherry Syrup or Cola.
    if 'Tomato Juice' in combination and ('Pineapple Juice' in combination or 'Coconut Cream' in combination or 'Lemonade' in combination or 'Worcestershire Sauce' in combination or 'Tabasco Sauce' in combination or 'Cherry Syrup' in combination or 'Cola' in combination):
        return False
    
    # Avoid combining Worcestershire Sauce, Tabasco Sauce, Cherry Syrup or Cola with Pineapple Juice, Coconut Cream or Lemonade.
    if ('Worcestershire Sauce' in combination or 'Tabasco Sauce' in combination or 'Cherry Syrup' in combination or 'Cola' in combination) and ('Pineapple Juice' in combination or 'Coconut Cream' in combination or 'Lemonade' in combination):
        return False
    
    # Do not combine Lemonade with Grapefruit Juice, Orange Juice or Cranberry Juice.
    if 'Lemonade' in combination and ('Grapefruit Juice' in combination or 'Orange Juice' in combination or 'Cranberry Juice' in combination):
        return False
    
    # Do not combine Orange Juice with Grapefruit Juice, Pineapple Juice or Cranberry Juice.
    if 'Orange Juice' in combination and ('Grapefruit Juice' in combination or 'Pineapple Juice' in combination or 'Cranberry Juice' in combination):
        return False
    
    # Avoid combining Mango or Sugar with any other ingredient.
    if ('Mango' in combination or 'Sugar' in combination) and len(combination) > 1:
        return False
    
    # Do not combine Lavender with Juice or Soda.
    if 'Lavender' in combination and ('Juice' in combination or 'Soda' in combination):
        return False
    
    # Do not combine Jalapeno with Grapefruit Juice, Pineapple Juice, or Lemonade.
    if 'Jalapeno' in combination and ('Grapefruit Juice' in combination or 'Pineapple Juice' in combination or 'Lemonade' in combination):
        return False
    
    # Do not combine Sage Leaves with Pineapple Juice, Grapefruit Juice, or Lemonade.
    if ('Sage Leaves' in combination) and ('Pineapple Juice' in combination or 'Grapefruit Juice' in combination or 'Lemonade' in combination):
        return False
    
    # Do not combine Cinnamon Stick or Star Anise with Juice or Soda.
    if ('Cinnamon Stick' in combination or 'Star Anise' in combination) and ('Juice' in combination or 'Soda' in combination):
        return False
    
    # Check if the combination contains both Grapefruit Juice and Club Soda or Passionfruit Juice.
    if 'Grapefruit Juice' in combination and ('Club Soda' in combination or 'Passionfruit Juice' in combination):
        return False
    
    # Check if the combination contains both Apple Cider and Cherry.
    if 'Apple Cider' in combination and 'Cherry' in combination:
        return False
    
    # Avoid using both banana and blueberries or raspberries together.
    if 'Banana' in combination and ('Blueberries' in combination or 'Raspberries' in combination):
        return False
    
    # Do not combine Jalapeno Syrup or Club Soda without Lime Juice.
    if ('Jalapeno Syrup' in combination or 'Club Soda' in combination) and 'Lime Juice' not in combination:
        return False
    
    # No combination should contain more than one of the following ingredients.
    ingredients_to_check = ['Ginger Ale', 'Passionfruit Juice', 'Sage Leaves', 'Maraschino Cherry',
                            'Jalapeno', 'Raspberries', 'Simple Syrup', 'Simple Syrup, Lime',
                            'Soda Water', 'Vanilla Extract']
    found_ingredients = [ingr for ingr in ingredients_to_check if ingr in combination]
    if len(found_ingredients) > 1:
        return False
    
    # Check for mutually exclusive ingredients
    mutually_exclusive_pairs = [("Honey", "Mint Leaves"), ("Honey", "Orange Peel"), ("Honey", "Spinach"),
                                ("Mint Leaves", "Orange Peel"), ("Mint Leaves", "Spinach"), ("Mint Leaves", "Agave Syrup"),
                                ("Beetroot Juice", "Club Soda"), ("Club Soda","Coconut Water"), ("Banana", "Orange Peel"), 
                                ("Orange Peel", "Spinach"), ("Spinach", "Banana"),('Agave Syrup', 'Sugar'), ('Agave Syrup', 'Sprite'), 
                                ('Agave Syrup', 'Honey'),('Agave Syrup', 'Simple Syrup, Lime'), ('Pineapple Juice', 'Sprite'),
                                ('Pineapple Juice', 'Simple Syrup, Lime'), ('Grapefruit Juice', 'Sprite'),
                                ('Grapefruit Juice', 'Simple Syrup, Lime'), ('Milk', 'Coconut Cream'),
                                ('Lime Juice', 'Soda Water'), ('Lime Juice', 'Tonic Water'),
                                ('Sage Leaves', 'Soda Water'), ('Sage Leaves', 'Tonic Water'),
                                ('Spinach', 'Soda Water'), ('Spinach', 'Tonic Water'), ('Jalapeno Syrup', 'Honey'), 
                                ('Jalapeno Syrup', 'Dandelion Root Tea'),('Honey', 'Dandelion Root Tea'), 
                                ('Strawberries', 'Dandelion Root Tea'),('Milk', 'Lemonade'), ('Milk', 'Lemon Lime Soda'), 
                                ('Milk', 'Club Soda'), ('Milk', 'Sprite'), ('Milk', 'Tonic Water'), 
                                ('Coconut Milk', 'Lemonade'), ('Coconut Milk', 'Lemon Lime Soda'), ('Coconut Milk', 'Club Soda'), 
                                ('Coconut Milk', 'Sprite'), ('Coconut Milk', 'Tonic Water'), ('Pineapple Juice', 'Lemonade'), 
                                ('Pineapple Juice', 'Lemon Lime Soda'), ('Pineapple Juice', 'Club Soda'), ('Pineapple Juice', 'Sprite'), 
                                ('Pineapple Juice', 'Tonic Water'), ('Orange Juice', 'Milk'), ('Orange Juice', 'Coconut Milk'), 
                                ('Orange Juice', 'Lemonade'), ('Orange Juice', 'Lemon Lime Soda'), ('Orange Juice', 'Club Soda'), 
                                ('Orange Juice', 'Sprite'), ('Orange Juice', 'Tonic Water'), ('Lemon Lime Soda', 'Milk'), 
                                ('Lemon Lime Soda', 'Coconut Milk'), ('Lemon Lime Soda', 'Pineapple Juice'), ('Lemonade', 'Milk'), 
                                ('Lemonade', 'Coconut Milk'), ('Lemonade', 'Pineapple Juice'), ('Club Soda', 'Milk'), 
                                ('Club Soda', 'Coconut Milk'), ('Club Soda', 'Pineapple Juice'), ('Sprite', 'Milk'), 
                                ('Sprite', 'Coconut Milk'), ('Sprite', 'Pineapple Juice'), ('Tonic Water', 'Milk'), 
                                ('Tonic Water', 'Coconut Milk'), ('Tonic Water', 'Pineapple Juice')]
    for pair in mutually_exclusive_pairs:
        if pair[0] in combination and pair[1] in combination:
            return False
    
    if ('Cucumber Juice' in combination and 'Grapefruit Juice' in combination) or \
           ('Cucumber Juice' in combination and 'Lemon Juice' in combination) or \
           ('Grapefruit Juice' in combination and 'Grapefruit Slice' in combination) or \
           ('Grapefruit Juice' in combination and 'Lemon Juice' in combination) or \
           ('Grapefruit Juice' in combination and 'Orange Juice' in combination) or \
           ('Lemon Juice' in combination and 'Orange Juice' in combination) or \
           ('Lemon Juice' in combination and 'Peach Tea' in combination) or \
           ('Lemon Juice' in combination and 'Lime Juice' in combination) or \
           ('Lemon Juice' in combination and 'Watermelon Juice' in combination):
            return False
    
    # If none of the above conditions are met, then the combination is feasible
    return True

def get_feasible_combinations(possible_combinations, preferred_flavors):
    feasible_combinations = []
    for combination in possible_combinations:
        if is_feasible(combination, preferred_flavors):
            feasible_combinations.append(combination)
    return feasible_combinations


possible_combinations = get_feasible_combinations(new_list, preferred_flavors)
print()
print("Feasible Combinations")
print("Total number of all possible combinations", len(possible_combinations))

# Get a list of all ingredients    
ingredients_1 = [i for i in df['Ingredient 1']]
ingredients_2 = [j for j in df['Ingredient 2']]
ingredients_3 = [k for k in df['Ingredient 3']]
all_ingredients = ingredients_1 + ingredients_2 + ingredients_3
all_ingredients = set(all_ingredients)

# Define the list of flavors
flavors_list = ['Sour', 'Sweet', 'Bitter', 'Creamy', 'Refreshing']

# Convert the dataset input into format for neural network
def row_to_input(row):
    ingredients = []
    flavors = []
    for i in range(1, 4):
        ingredient = row[f'Ingredient {i}']
        if pd.notna(ingredient):
            ingredients.append(ingredient)
            if ingredient in all_ingredients:
                ingredient_df = df[df[['Ingredient 1', 'Ingredient 2', 'Ingredient 3']].isin([ingredient]).any(axis=1)]
                flavor = ingredient_df['Flavor Profile 1'].values[0]
                flavors.append([1 if f in flavor else 0 for f in flavors_list])
            else:
                flavors.append([0] * len(flavors_list))
    if len(ingredients) == 0:
        return None
    else:
        return np.array(flavors).flatten()

# Split the data into training and testing data sets
train_df = df.sample(frac=0.8, random_state=123)
test_df = df.drop(train_df.index)

# Convert the training and testing dataframes into TensorFlow inputs
train_inputs = [row_to_input(row) for _, row in train_df.iterrows() if row_to_input(row) is not None]
train_inputs = np.vstack(train_inputs)
train_targets = np.array(train_df['User Rating'])

# Define the model architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_inputs.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
# Train the model with early stopping
model.fit(train_inputs, train_targets, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the performance of the model on the testing data
test_inputs = [row_to_input(row) for _, row in test_df.iterrows() if row_to_input(row) is not None]
test_inputs = np.vstack(test_inputs)
test_targets = np.array(test_df['User Rating'])
test_loss = model.evaluate(test_inputs, test_targets)
print('Test loss:', test_loss)

# Get the flavor profiles of the new cocktail from the user
new_flavors = [flavour_1,flavour_2]

# Use neural network to predict the user rating of each possible mocktail
best_rating = 0
best_combination = possible_combinations[0]
for combination in possible_combinations:
    # Convert the flavor profiles into the same format as the training data
    new_input = row_to_input({'Ingredient 1': combination[0], 'Ingredient 2': combination[1], 'Ingredient 3': combination[2], 'Flavor Profile 1': new_flavors[0], 'Flavor Profile 2': new_flavors[1]})
    # Pass the converted input to the trained model
    rating = model.predict(np.array([new_input]))[0][0]
    if rating > best_rating:
        best_rating = rating
        print(rating)
        best_combination = combination
        print(combination)
        
print("Best combination:", best_combination)
print("Best rating:", best_rating)

# Naming the mocktail
flavor_profile_adjectives = {
    "Sweet": ['Delicious', 'Divine', 'Heavenly', 'Luscious', 'Yummy', 'Scrumptious', 'Tasty', 'Delectable'],
    "Sour": ['Tangy', 'Zesty', 'Sour', 'Puckery', 'Sharp', 'Lively', 'Fresh'],
    "Bitter": ['Bold', 'Robust', 'Intense', 'Earthy', 'Smokey', 'Complex', 'Strong'],
    "Refreshing": ['Cool', 'Crisp', 'Invigorating', 'Refreshing', 'Revitalizing', 'Soothing'],
    "Creamy": ['Creamy', 'Smooth', 'Silky', 'Luscious', 'Velvety', 'Rich']
}

flavor_nouns = {
    "Sweet": ['Bliss', 'Crush', 'Dream', 'Delight', 'Sweetheart'],
    "Sour": ['Lemonade', 'Patch', 'Spark', 'Sensation', 'Tingle'],
    "Bitter": ['Bite', 'Kick', 'Symphony'],
    "Refreshing": ['Breeze', 'Splash', 'Chill', 'Refresher', 'Revive', 'Rejuvenation', 'Mist'],
    "Creamy": ['Dream', 'Temptation', 'Refresher']
}

ingredient_nouns = {
  "Lime Juice": ["Tang", "Zest", "Citrus", "Limeade"],
  "Orange Juice": ["Citrus", "Orange Blossom", "Sunny Delight", "Juice"],
  "Agave Syrup": ["Sweetener", "Nectar", "Cactus Syrup", "Honey"],
  "Lemon Lime Soda": ["Fizz", "Citrus Bubbles", "Sprite", "Carbonated Water"],
  "Grenadine": ["Pomegranate Syrup", "Sweet Red", "Ruby", "Red Syrup"],
  "Maraschino Cherry": ["Red Cherry", "Sweet Cherry", "Luxardo", "Cherry Bomb"],
  "Pineapple Juice": ["Tropic", "Pineapple Paradise", "Juicy Pineapple", "Pineapple Express"],
  "Coconut Cream": ["Cream of Coconut", "Coconut Milk", "Coconut Heaven", "Coconut Dream"],
  "Pineapple Slice": ["Pineapple Spear", "Pineapple Wedge", "Pineapple Ring", "Pineapple Garnish"],
  "Mint Leaves": ["Fresh Mint", "Cool Mint", "Mint Sprig", "Minty"],
  "Soda Water": ["Sparkling Water", "Bubbly", "Fizzy", "Carbonated Water"],
  "Strawberries": ["Berry", "Sweetheart", "Strawberry Fields", "Red Delicious"],
  "Simple Syrup": ["Sweet Syrup", "Sugar Syrup", "Clear Syrup", "Sweetener"],
  "Tomato Juice": ["Tomatoey", "Savory Juice", "Bloody Mary Mix", "Spicy V8"],
  "Worcestershire Sauce": ["Worcestershire", "Savory Sauce", "Umami Sauce", "Bold Sauce"],
  "Tabasco Sauce": ["Hot Sauce", "Spicy Sauce", "Pepper Sauce", "Fiery Sauce"],
  "Cherry Syrup": ["Cherry Juice", "Sweet Cherry", "Cherry Cola", "Cherry Bomb"],
  "Cola": ["Soda", "Coke", "Classic", "Fizzy Drink"],
  "Lemonade": ["Lemon Juice", "Lemon Refresher", "Citrus Cooler", "Lemon Splash"],
  "Club Soda": ["Sparkling Water", "Bubbly", "Fizzy", "Carbonated Water"],
  "Peach Syrup": ["Peach Juice", "Sweet Peach", "Peach Fuzz", "Peachy"],
  "Peach Slice": ["Peach Wedge", "Peach Garnish", "Peachy Keen", "Peach Perfection"],
  "Blackberries": ["Berry", "Blackberry Bliss", "Sweetheart", "Dark Delight"],
  "Blueberries": ["Berry", "Blueberry Burst", "Sweetheart", "Blue Crush"],
  "Banana": ["Banana Cream", "Banana Split", "Banana Blast", "Sweetheart"],
  "Yogurt": ["Creamy", "Smoothie", "Yogurt Delight", "Yogurt Parfait"],
  "Honey": ["Sweetener", "Nectar", "Liquid Gold", "Honeycomb"],
  "Spinach": ["Leafy Green", "Green Goodness", "Popeye's Favorite", "Healthy Greens"],
  "Cranberry Juice": ["Tart Juice", "Cranberry Blast", "Cranberry Splash", "Ruby Red"],
}

flavor_profile = preferred_flavors[0]
ingredients = best_combination

adjectives = flavor_profile_adjectives[flavor_profile]
for ingredient in ingredients:
    if ingredient in ingredient_nouns:
        nouns = ingredient_nouns[ingredient]
        flavor_nouns[flavor_profile].extend(nouns)

adjective = random.choice(adjectives)
noun = random.choice(flavor_nouns[flavor_profile])

mocktail_name = f"{adjective} {noun} Mocktail"
        
if best_combination is not None:
    print(f"I suggest {mocktail_name}. This is made of {best_combination[0]}, {best_combination[1]}, and {best_combination[2]}. Enjoy!")
else:
    print("Sorry, we couldn't find a mocktail that satisfies your preferences.")


if __name__ == '__main__':
    app.run(debug=True, port=4040)