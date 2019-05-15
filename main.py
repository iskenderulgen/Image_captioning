import os
from pickle import dump
from pickle import load

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from Captioning_Model.Caption_Generator import generate_desc, extract_features_caption
from Captioning_Model.Description_Extractor import load_doc, load_descriptions, clean_descriptions, to_vocabulary, \
    save_descriptions
from Captioning_Model.Extract_Features import extract_features
from Captioning_Model.Test import evaluate_model
from Captioning_Model.Train import load_set, load_clean_descriptions, load_photo_features, create_tokenizer, max_length, \
    create_sequences, define_model

PATH = "/home/ulgen/Documents/Pycharm_Workspace/Caption_VGG/Data/"
Script_Path = "/home/ulgen/Documents/Pycharm_Workspace/Caption_VGG/Summary_Model/"
model_eval = "no"
# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
if __name__ == "__main__":

    # This seciton extracts the Features from pictures
    exists = os.path.isfile(PATH + "Outputs/features.pkl")
    if exists:
        print("File Already Exist")
    else:
        # In This Part we do feautre extraction using VGG16
        features = extract_features(PATH + "Flicker8k_Dataset")
        print("Feature Extraction Finished")
        print('Extracted Features: %d' % len(features))
        # save to file
        dump(features, open(PATH + 'Outputs/features.pkl', 'wb'))

    # This section extract descriptions
    exists = os.path.isfile(PATH + 'Outputs/descriptions.txt')
    if exists:
        print("File Already Exist")
    else:
        filename = PATH + 'Flickr8k_text/Flickr8k.token.txt'
        # load descriptions
        doc = load_doc(filename)
        # parse descriptions
        descriptions = load_descriptions(doc)
        print('Loaded: %d ' % len(descriptions))
        # clean descriptions
        clean_descriptions(descriptions)
        # summarize vocabulary
        vocabulary = to_vocabulary(descriptions)
        print('Vocabulary Size: %d' % len(vocabulary))
        # save to file
        save_descriptions(descriptions, PATH + 'Outputs/descriptions.txt')

    # This section trains the model
    exists = os.path.isfile(PATH + 'Outputs/model.h5')
    if exists:
        print("File Already Exist")
    else:
        # train dataset

        # load training dataset (6K)
        filename = PATH + 'Flickr8k_text/Flickr_8k.trainImages.txt'
        train = load_set(filename)
        print('Dataset: %d' % len(train))
        # descriptions
        train_descriptions = load_clean_descriptions(PATH + 'Outputs/descriptions.txt', train)
        print('Descriptions: train=%d' % len(train_descriptions))
        # photo features
        train_features = load_photo_features(PATH + 'Outputs/features.pkl', train)
        print('Photos: train=%d' % len(train_features))
        # prepare tokenizer
        tokenizer = create_tokenizer(train_descriptions)
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % vocab_size)
        # determine the maximum sequence length
        max_length_res = max_length(train_descriptions)
        print('Description Length: %d' % max_length_res)
        # prepare sequences
        X1train, X2train, ytrain = create_sequences(tokenizer, max_length_res, train_descriptions, train_features,
                                                    vocab_size)

        # dev dataset

        # load test set
        filename = PATH + 'Flickr8k_text/Flickr_8k.devImages.txt'
        test = load_set(filename)
        print('Dataset: %d' % len(test))
        # descriptions
        test_descriptions = load_clean_descriptions(PATH + 'Outputs/descriptions.txt', test)
        print('Descriptions: test=%d' % len(test_descriptions))
        # photo features
        test_features = load_photo_features(PATH + 'Outputs/features.pkl', test)
        print('Photos: test=%d' % len(test_features))
        # prepare sequences
        X1test, X2test, ytest = create_sequences(tokenizer, max_length_res, test_descriptions, test_features,
                                                 vocab_size)

        # fit model

        # define the model
        model = define_model(vocab_size, max_length_res)
        # define checkpoint callback
        filepath = PATH + 'Outputs/model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # fit model
        model.fit([X1train, X2train], ytrain, epochs=2, verbose=2, callbacks=[checkpoint],
                  validation_data=([X1test, X2test], ytest))

    if model_eval == "yes":
        # prepare tokenizer on train set

        # load training dataset (6K)
        filename = PATH + 'Flickr8k_text/Flickr_8k.trainImages.txt'
        train = load_set(filename)
        print('Dataset: %d' % len(train))
        # descriptions
        train_descriptions = load_clean_descriptions(PATH + 'Outputs/descriptions.txt', train)
        print('Descriptions: train=%d' % len(train_descriptions))
        # prepare tokenizer
        tokenizer = create_tokenizer(train_descriptions)
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % vocab_size)
        # determine the maximum sequence length
        max_length_res = max_length(train_descriptions)
        print('Description Length: %d' % max_length_res)

        # prepare test set

        # load test set
        filename = PATH + 'Flickr8k_text/Flickr_8k.testImages.txt'
        test = load_set(filename)
        print('Dataset: %d' % len(test))
        # descriptions
        test_descriptions = load_clean_descriptions(PATH + 'Outputs/descriptions.txt', test)
        print('Descriptions: test=%d' % len(test_descriptions))
        # photo features
        test_features = load_photo_features(PATH + 'Outputs/features.pkl', test)
        print('Photos: test=%d' % len(test_features))

        # load the model
        filename = PATH + 'Outputs/model.h5'
        model = load_model(filename)
        # evaluate model
        evaluate_model(model, test_descriptions, test_features, tokenizer, max_length_res)

    exists = os.path.isfile(PATH + 'Outputs/tokenizer.pkl')
    if exists:
        print("File Already Exist")
    else:
        # load training dataset (6K)
        filename = PATH + 'Flickr8k_text/Flickr_8k.trainImages.txt'
        train = load_set(filename)
        print('Dataset: %d' % len(train))
        # descriptions
        train_descriptions = load_clean_descriptions(PATH + 'Outputs/descriptions.txt', train)
        print('Descriptions: train=%d' % len(train_descriptions))
        # prepare tokenizer
        tokenizer = create_tokenizer(train_descriptions)
        # save the tokenizer
        dump(tokenizer, open(PATH + 'Outputs/tokenizer.pkl', 'wb'))

    twilight_captions = list()

    # load the tokenizer
    tokenizer = load(open(PATH + 'Outputs/tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length_res = 34
    # load the model
    model = load_model(PATH + 'Outputs/model.h5')
    # load and prepare the photograph
    counter = 0
    for filename in os.listdir(PATH + "Extracted_Images"):
        photo = extract_features_caption(os.path.join(PATH + "Extracted_Images/", filename))
        description = generate_desc(model, tokenizer, photo, max_length_res)
        print(description)
        twilight_captions.append(description)
        print(counter)
        counter += 1

    with open(PATH + 'Outputs/twi_caps.txt', 'w') as f:
        for item in twilight_captions:
            f.write("%s\n" % item)
