import tensorflow as tf

def create_callbacks(model_checkpoint_path='model_checkpoint.h5', log_dir='logs', early_stopping_patience=2):
    # Model checkpoint callback
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True)
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, monitor='val_loss')
    
    # TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    return [model_checkpoint, early_stopping, tensorboard]
