"""
Email Auto-Responder Simple Reflex Agent benchmark for AgenticEvals.

This benchmark tests a model's ability to provide immediate, rule-based responses
to email scenarios based on keywords and patterns, without requiring memory or planning.
"""

import time
import re
from typing import List, Dict, Any
from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@benchmark(
    name="email_autoresponder_simple",
    agent_type=AgentType.SIMPLE_REFLEX,
    description="Email auto-responder benchmark testing immediate rule-based responses to email scenarios"
)
class EmailAutoresponderBenchmark(BaseBenchmark):
    """
    Simple reflex agent benchmark using email auto-responder scenarios.
    
    Tests the model's ability to provide immediate, rule-based responses
    to email content based on keywords and patterns.
    """
    
    def get_tasks(self) -> List[Task]:
        """Get all email auto-responder tasks."""
        tasks = []
        
        # Realistic challenging email scenarios
        challenging_scenarios = [
            # 1. Billing with urgency mixed in
            ("billing_response", "billing_urgent_mix", """
Subject: URGENT: Invoice #A-2024-0892 Discrepancy - Need Immediate Resolution

Dear Billing Team,

I hope this message finds you well. I'm writing regarding invoice #A-2024-0892 dated March 15th, 2024, which shows a charge of $2,847.50 for our annual software license renewal.

However, I have several concerns that need urgent attention:

1. The invoice amount doesn't match our agreed-upon rate of $2,200 from our contract signed in January
2. We were supposed to receive a 15% discount for early payment, but this doesn't appear to be reflected
3. Our finance team needs this resolved by EOD Friday as they're closing the books for Q1
4. We also noticed an additional "processing fee" of $97.50 that wasn't mentioned in our original agreement

I've attached our signed contract and previous correspondence for your review. Given the time-sensitive nature of this matter, could you please prioritize this and get back to me by tomorrow morning?

We value our partnership and want to get this sorted out quickly. Please let me know if you need any additional documentation.

Best regards,
Sarah Martinez
CFO, TechCorp Solutions
"""),
            
            # 2. Meeting with support elements
            ("calendar_response", "meeting_support_mix", """
Subject: Rescheduling Tuesday's Product Demo + Training Session Questions

Hi Team,

I need to reschedule our Tuesday 2PM product demonstration meeting due to a client emergency. 

Originally, we were planning to cover:
- Q4 feature roadmap review
- New dashboard functionality walkthrough  
- Training on the advanced reporting module
- Q&A session with the development team

However, I've been getting several support tickets from our users about the new interface changes, and I think it would be helpful to address these during our session as well. Some of the common issues include:

1. Users can't find the export function in the new layout
2. The search functionality seems slower than before
3. Several people are asking for help with the new keyboard shortcuts

Could we move this to Thursday same time? Also, should we invite someone from customer support to help address these user concerns during the demo?

Please confirm your availability. I know this is short notice, but I want to make sure we have the most productive session possible.

Thanks,
Mike Chen
Product Manager
"""),
            
            # 3. Security with billing undertones
            ("security_response", "security_billing_mix", """
Subject: Suspicious Payment Activity on Account #4456 - Password Reset Required

Dear Security Team,

I'm contacting you regarding some concerning activity on our company account #4456. 

This morning, our accounting department noticed several unauthorized payment attempts on our corporate credit card ending in 8732. The attempts were made between 2:30 AM and 3:15 AM PST, which is well outside our normal business hours.

The concerning part is that these payment attempts were made through our customer portal, which suggests someone may have gained access to our account credentials. The failed transactions total $3,247.89 and appear to be for services we never ordered.

Immediate actions needed:
1. Reset password for account #4456
2. Review all recent login activity for suspicious patterns
3. Check if any of our billing information has been modified
4. Investigate how the unauthorized access occurred

I've already contacted our bank to freeze the card, but we need to secure our account immediately. Our CEO is very concerned about this security breach, especially since we handle sensitive client data.

Can someone from your team call me at 555-0123 as soon as possible? We need to resolve this before our board meeting at 4 PM today.

Urgently,
David Rodriguez
IT Director, SecureLogistics Inc.
"""),
            
            # 4. Support with meeting elements
            ("support_response", "support_meeting_mix", """
Subject: Help Needed: Integration Issues Before Tomorrow's Client Presentation

Hi Support Team,

I'm reaching out because we're experiencing some technical difficulties with our API integration, and we have a critical client presentation tomorrow morning at 9 AM.

The issue: Our data synchronization between your platform and our CRM system (Salesforce) has been failing intermittently since last Friday. We're getting error code 429 (too many requests) even though we're well within our API rate limits according to our dashboard.

This is particularly problematic because:
- We need to demonstrate real-time data updates to our biggest client
- The presentation is for a potential $500K contract renewal
- Our sales team has been preparing for this meeting for weeks

I've already tried:
- Clearing cache and cookies
- Testing with different API keys  
- Reducing our request frequency
- Checking our integration logs

What I need from you:
1. Help troubleshooting the API connection issues
2. Possibly a quick screen-share session to review our setup
3. Confirmation that our integration will be stable for tomorrow's demo

Is there someone available for a brief call this afternoon? I can make myself available anytime between 2-6 PM EST. If not, could we schedule an early morning troubleshooting session before the presentation?

This is really urgent for our business relationship with this client.

Thanks in advance,
Lisa Thompson
Technical Sales Lead
"""),
            
            # 5. Refund with urgent priority
            ("billing_response", "refund_urgent_mix", """
Subject: URGENT REFUND REQUEST - Duplicate Charges on Account

Dear Billing Department,

I'm writing to request an immediate refund for duplicate charges that appeared on our account statement this morning.

The situation: We were charged twice for the same service upgrade on March 22nd:
- Charge 1: $1,299.00 at 9:23 AM (Transaction ID: TXN-8847392)
- Charge 2: $1,299.00 at 9:24 AM (Transaction ID: TXN-8847401)

This appears to be a system error during the checkout process. I remember the payment page seemed to freeze after I clicked "Submit," so I may have accidentally clicked it twice.

Why this is urgent:
- Our monthly budget reconciliation is due tomorrow
- Finance needs to report accurate numbers to our board
- This double charge is affecting our cash flow for vendor payments due this week

I've attached screenshots of both charges and our account statement. The duplicate charge needs to be refunded to our original payment method (corporate card ending in 9876).

Could you please:
1. Process the refund for the duplicate charge immediately
2. Send me a confirmation email with the refund transaction details
3. Investigate why the system allowed duplicate charges so quickly

I need this resolved by end of business today. Please contact me at 555-0167 if you need any additional information.

Thank you for your prompt attention to this matter.

Sincerely,
Robert Kim
Finance Manager
"""),
            
            # 6. Complex multi-category email
            ("security_response", "complex_multi_category", """
Subject: Account Lockout + Meeting Cancellation + Billing Question

Hi,

I'm dealing with multiple issues that started this morning and need help prioritizing what to address first.

MAIN ISSUE - Account Security:
My login credentials stopped working around 8 AM, and I'm getting a message saying "Account temporarily locked due to suspicious activity." I haven't shared my password with anyone, and I was logged in fine yesterday. I need access restored ASAP because I have client work due today.

RELATED ISSUE - Meeting Impact:
Because I can't access the system, I had to cancel my 11 AM demo with a potential client. This is embarrassing and potentially costly. Can we reschedule for Friday, and will my access definitely be restored by then?

BILLING CONCERN:
While trying to understand what happened, I noticed we were charged $89 for "additional security monitoring" this month. Is this related to the account lockout? We weren't notified about this charge, and I don't remember authorizing it.

URGENT QUESTIONS:
1. How do I get my account unlocked immediately?
2. What caused the suspicious activity trigger?
3. Should we be concerned about a security breach?
4. Can we get a refund for the security monitoring fee if it's related to a false alarm?
5. Who should I contact about rescheduling client meetings affected by this?

This is affecting our business operations, so please treat this as high priority.

Thanks,
Jennifer Walsh
"""),
            
            # 7. Password reset with billing context
            ("security_response", "password_billing_context", """
Subject: Password Reset Request - Unable to Access Billing Portal

Dear Support,

I need to reset my password for our company account. I haven't been able to log into the billing portal for the past three days, and our payment for this month's services is due tomorrow.

Background:
- My usual password isn't working (I haven't changed it recently)
- The "Forgot Password" link sends me to a page that just says "Service temporarily unavailable"
- I need to update our payment method from the old credit card to our new corporate card
- Our automatic payment failed yesterday, and I'm worried about service interruption

The tricky part is that our previous IT administrator (who set up the account) left the company last month, and we don't have access to the original recovery email address (it was his personal email).

Can you help me:
1. Reset the password for account holder "jennifer.walsh@techinnovate.com"
2. Update the recovery email to my current business email
3. Ensure we can process payment before the service deadline
4. Confirm our services won't be interrupted while we sort this out

I can provide company documentation, tax ID, or other verification if needed. Just please help me get this resolved before we face any service disruption.

Thank you,
Jennifer Walsh
Acting IT Manager
TechInnovate Solutions
"""),
            
            # 8. Support with meeting scheduling
            ("support_response", "support_calendar_priority", """
Subject: Integration Training Session + Technical Support Needed

Hello,

Our team needs help with two related issues:

PRIMARY ISSUE - Technical Support:
We're struggling with the new API integration and need hands-on technical support. Our developer has been working on this for two weeks, but we're still getting authentication errors when trying to connect our inventory management system.

Error messages we're seeing:
- "Invalid client credentials" (even though we're using the correct API key)
- "Rate limit exceeded" (but we're only making 10 requests per minute)
- "Malformed request" (our JSON structure looks correct to us)

SECONDARY ISSUE - Training Request:
Given these technical challenges, we'd like to schedule a comprehensive training session with your integration specialists. Our whole development team (4 people) needs to understand:
- Best practices for API authentication
- Proper error handling and retry logic
- How to optimize our request patterns
- Advanced integration techniques

Could we schedule a 2-hour technical workshop sometime next week? We're flexible on timing, but would prefer:
- Tuesday or Wednesday afternoons
- Virtual meeting (we can provide Zoom or Teams)
- Screen sharing capability so we can walk through our code together

This integration is crucial for our Q2 product launch, so any help you can provide would be greatly appreciated.

Please let me know your availability and what information you need from us to prepare for the session.

Best regards,
Alex Morgan
Lead Developer
"""),
        ]
        
        # Create tasks from challenging scenarios
        for i, (expected_response, scenario_name, email_content) in enumerate(challenging_scenarios):
            task = Task(
                task_id=f"email_challenge_{i+1}",
                name=f"Complex Email: {scenario_name}",
                description=f"Challenging email classification requiring careful analysis of mixed signals",
                prompt=self._create_advanced_prompt(email_content.strip()),
                expected_output=expected_response,
                evaluation_criteria={
                    "exact_match": True,
                    "case_insensitive": True,
                    "keywords": [expected_response]
                },
                metadata={
                    "scenario": scenario_name,
                    "expected_response": expected_response,
                    "difficulty": "challenging",
                    "email_length": len(email_content.split()),
                    "complexity": "high"
                }
            )
            tasks.append(task)
        
        return tasks
    
    def _create_advanced_prompt(self, email_content: str) -> str:
        """Create an advanced prompt for complex email classification."""
        return f"""Advanced Email Auto-Responder Classification:

You are an intelligent email classification system for a business. You must analyze the email content and classify it into exactly ONE of these categories:

CATEGORIES:
- "billing_response" - Issues related to invoices, payments, refunds, charges, pricing, or financial matters
- "security_response" - Issues related to passwords, account access, login problems, suspicious activity, or security breaches  
- "calendar_response" - Issues related to meetings, scheduling, appointments, demos, or time coordination
- "support_response" - Technical issues, product help, troubleshooting, training, or general assistance
- "priority_response" - General urgent matters that don't fit the above categories

CLASSIFICATION RULES:
1. If multiple categories apply, choose the PRIMARY concern based on what needs immediate action
2. Security issues (passwords, breaches, locked accounts) take highest priority
3. Billing issues (payment problems, refunds, charges) take second priority  
4. Support issues (technical problems, training) take third priority
5. Calendar issues (scheduling, meetings) take fourth priority
6. Use "priority_response" only for urgent matters that don't fit other categories

INSTRUCTIONS:
- Read the entire email carefully
- Identify the main issue that needs to be addressed
- Respond with exactly one category name
- Do not provide explanations or additional text

EMAIL TO CLASSIFY:
{email_content}

CLASSIFICATION:"""
    
    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single email auto-responder task."""
        start_time = time.time()
        
        try:
            # Generate model response
            model_response = await model.generate(task.prompt)
            execution_time = time.time() - start_time
            
            # Calculate score
            score = self.calculate_score(task, model_response)
            success = score > 0.5
            
            # Calculate detailed metrics
            metrics = self._calculate_detailed_metrics(task, model_response)
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=success,
                score=score,
                metrics=metrics,
                model_response=model_response,
                execution_time=execution_time,
                metadata=task.metadata
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,
                score=0.0,
                metrics={},
                execution_time=execution_time,
                error_message=str(e),
                metadata=task.metadata
            )
    
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Calculate score for an email auto-responder task."""
        expected = task.expected_output.lower() if task.expected_output else ""
        response_text = model_response.text.strip().lower() if model_response.text else ""
        
        # Remove common punctuation and whitespace
        response_text = re.sub(r'[.,!?;:\s]', '', response_text)
        expected = re.sub(r'[.,!?;:\s]', '', expected)
        
        # Exact match gets full score
        if response_text == expected:
            return 1.0
        
        # Check if expected response is contained in the response
        if expected in response_text:
            return 0.8
        
        # Check for alternative valid response formats
        valid_responses = [
            "billing_response", "billingresponse", "billing",
            "priority_response", "priorityresponse", "priority", "urgent",
            "calendar_response", "calendarresponse", "calendar", "meeting",
            "support_response", "supportresponse", "support",
            "security_response", "securityresponse", "security", "password"
        ]
        
        # Map expected to alternatives
        expected_alternatives = {
            "billing_response": ["billingresponse", "billing"],
            "priority_response": ["priorityresponse", "priority", "urgent"],
            "calendar_response": ["calendarresponse", "calendar", "meeting"],
            "support_response": ["supportresponse", "support"],
            "security_response": ["securityresponse", "security", "password"]
        }
        
        if expected in expected_alternatives:
            for alternative in expected_alternatives[expected]:
                if alternative in response_text:
                    return 0.7
        
        return 0.0
    
    def _calculate_detailed_metrics(self, task: Task, model_response: ModelResponse) -> Dict[str, Any]:
        """Calculate detailed metrics for analysis."""
        response_text = model_response.text.strip() if model_response.text else ""
        expected = task.expected_output.lower() if task.expected_output else ""
        
        # Response analysis
        word_count = len(response_text.split())
        char_count = len(response_text)
        contains_expected = expected in response_text.lower() if expected and response_text else False
        
        # Check if response follows instructions (should be one response type)
        follows_instructions = word_count <= 2  # Allow for some flexibility
        
        # Extract actual response
        cleaned_response = re.sub(r'[.,!?;:\s]', '', response_text.lower())
        
        # Check if response is a valid response type
        valid_responses = [
            "billing_response", "billingresponse", "billing",
            "priority_response", "priorityresponse", "priority", "urgent",
            "calendar_response", "calendarresponse", "calendar", "meeting",
            "support_response", "supportresponse", "support",
            "security_response", "securityresponse", "security", "password"
        ]
        
        is_valid_response = any(valid in cleaned_response for valid in valid_responses)
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "follows_instructions": follows_instructions,
            "contains_expected": contains_expected,
            "is_valid_response": is_valid_response,
            "cleaned_response": cleaned_response,
            "exact_match": cleaned_response == expected.replace("_", "").replace(" ", ""),
            "response_latency": model_response.latency,
            "tokens_used": model_response.total_tokens,
        } 