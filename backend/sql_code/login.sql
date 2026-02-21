-- This SQL file represents the query used for the login endpoint.
-- It fetches the user's details and password hash using their email.

SELECT
    id,
    name,
    email,
    phone,
    password_hash,
    created_at,
    updated_at
FROM
    users
WHERE
    email = 'user_email_here';
