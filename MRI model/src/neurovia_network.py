"""
3D CNN Architecture for Alzheimer's Disease Classification
"""

def create_neurovia_model(
    input_dims=[None, None, None], 
    output_classes=1, 
    depth=5, 
    base_filters=4, 
    conv_l2=0.0, 
    dense_l2=1.0,
    learning_rate=0.0001, 
    trainable=True, 
    global_pool=False, 
    regression=False, 
    loss_fn='mae'
):
    
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, GlobalAveragePooling3D, 
                             Flatten, Dense, Dropout, Activation, BatchNormalization)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import backend as K
    
    # Input layer based on data format
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
        inputs = Input(shape=(1, input_dims[0], input_dims[1], input_dims[2]))
    else:
        bn_axis = -1
        inputs = Input(shape=(input_dims[0], input_dims[1], input_dims[2], 1))

    # Block 1: Base feature extraction
    x = Conv3D(base_filters, (3, 3, 3), activation='relu', padding='same', 
               trainable=trainable, kernel_regularizer=l2(conv_l2))(inputs)
    x = Conv3D(base_filters, (3, 3, 3), activation=None, padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.99, epsilon=0.001)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(x)

    # Block 2: Feature enhancement
    x = Conv3D(base_filters*2, (3, 3, 3), activation='relu', padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = Conv3D(base_filters*2, (3, 3, 3), activation=None, padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.99, epsilon=0.001)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(x)

    # Block 3: Deep feature extraction
    x = Conv3D(base_filters*4, (3, 3, 3), activation='relu', padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = Conv3D(base_filters*4, (3, 3, 3), activation=None, padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.99, epsilon=0.001)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(x)

    # Block 4: High-level features
    x = Conv3D(base_filters*8, (3, 3, 3), activation='relu', padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = Conv3D(base_filters*8, (3, 3, 3), activation=None, padding='same',
               trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.99, epsilon=0.001)(x)
    x = Activation('relu')(x)
    
    if global_pool and depth == 4:
        x = GlobalAveragePooling3D()(x)
    else:
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(x)

    # Optional Block 5: Complex features
    if depth >= 5:
        x = Conv3D(base_filters*16, (3, 3, 3), activation='relu', padding='same',
                   trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
        x = Conv3D(base_filters*16, (3, 3, 3), activation=None, padding='same',
                   trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
        x = BatchNormalization(axis=bn_axis, momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        
        if global_pool:
            x = GlobalAveragePooling3D()(x)
        else:
            x = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(x)

    # Optional Block 6: Very deep features
    if depth == 6:
        x = Conv3D(base_filters*32, (3, 3, 3), activation='relu', padding='same',
                   trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
        x = Conv3D(base_filters*32, (3, 3, 3), activation=None, padding='same',
                   trainable=trainable, kernel_regularizer=l2(conv_l2))(x)
        x = BatchNormalization(axis=bn_axis, momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        
        if global_pool:
            x = GlobalAveragePooling3D()(x)
        else:
            x = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(x)

    # Flatten if not using global pooling
    if not global_pool:
        x = Flatten()(x)

    # Output layer
    if not regression:
        if output_classes == 1:
            output = Dense(output_classes, activation='sigmoid', 
                          kernel_regularizer=l2(dense_l2), use_bias=True)(x)
        else:
            output = Dense(output_classes, activation='softmax',
                          kernel_regularizer=l2(dense_l2), use_bias=True)(x)
    else:
        output = Dense(output_classes, activation=None,
                      kernel_regularizer=l2(dense_l2))(x)

    # Create and compile model
    model = Model(inputs=inputs, outputs=output)
    
    # Use modern optimizer with learning_rate parameter
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy' if output_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"NeuraVia model created with learning rate: {learning_rate}")
    return model
