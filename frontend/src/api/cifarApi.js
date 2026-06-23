const BASE_URL = import.meta.env.VITE_API_URL || '/api';

/**
 * Predict the class of an image file using both ANN and CNN models.
 * @param {File} file - The image file to predict.
 * @returns {Promise<Object>} The prediction results.
 */
export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Prediction failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Error in predictImage:', error);
    throw error;
  }
};

/**
 * Fetch the final comparison metrics between ANN and CNN.
 * @returns {Promise<Object>} The comparison metrics.
 */
export const getComparison = async () => {
  try {
    const response = await fetch(`${BASE_URL}/comparison`);

    if (!response.ok) {
      throw new Error('Failed to fetch comparison metrics');
    }

    return await response.json();
  } catch (error) {
    console.error('Error in getComparison:', error);
    throw error;
  }
};

/**
 * Fetch misclassified samples for a specific model.
 * @param {string} model - The model type ('ann' or 'cnn').
 * @returns {Promise<Object>} The misclassified samples data.
 */
export const getMisclassified = async (model) => {
  try {
    const response = await fetch(`${BASE_URL}/misclassified/${model}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch misclassified samples for ${model}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error in getMisclassified(${model}):`, error);
    throw error;
  }
};

/**
 * Check the health status of the backend API and loaded models.
 * @returns {Promise<Object>} The health status results.
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(`${BASE_URL}/health`);

    if (!response.ok) {
      throw new Error('Backend health check failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Error in checkHealth:', error);
    throw error;
  }
};
